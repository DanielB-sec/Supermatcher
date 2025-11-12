"""
Biometric Worker Functions for ProcessPool
Isolated worker functions for CPU-bound biometric operations.

These functions are designed to run in separate processes via ProcessPoolExecutor.
They import supermatcher_v1_0 INSIDE the worker to avoid pickling issues.

MULTIPROCESSING STRATEGY:
=========================

1. VERIFY (1:1 matching): NO multiprocessing
   - Single comparison: probe vs 1 enrolled template
   - Fast operation (~2-3s), multiprocessing overhead > benefit
   - Server-level parallelism: Multiple concurrent verify requests use different workers

2. IDENTIFY (1:N matching): ADAPTIVE multiprocessing
   - Small galleries (<20 users): Sequential matching (low overhead)
   - Large galleries (≥20 users): Parallel matching
     * Probe extracted once
     * Gallery split into chunks (1 per CPU core)
     * Each chunk processed in parallel
     * Results merged and sorted
   - Speedup: ~4-8x on 8-core CPU for 100+ users

3. ADD_USER (enrollment): NO multiprocessing within single user
   - Single user with multiple images
   - Fusion algorithm already optimized
   - Server-level parallelism: Multiple add_user requests use different workers

4. LOAD_FOLDER (batch enrollment): YES multiprocessing
   - Processes N users in parallel (1 user per CPU core)
   - Each user enrolled independently
   - Speedup: ~4-8x on 8-core CPU for 10+ users

"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Any
import pickle
import sys


def worker_match_probe_against_gallery_chunk(
    probe_secure_dict: Dict[str, Any],
    gallery_chunk_secure_dicts: Dict[str, Dict[str, Any]],
    hasher_params: dict
) -> List[Tuple[str, float]]:
    """
    Worker function to match a probe against a chunk of gallery templates.
    Used for parallel identification.
    
    Args:
        probe_secure_dict: Secure dict of probe FingerprintTemplate
        gallery_chunk_secure_dicts: Dict mapping user_id -> secure dict (subset of gallery)
        hasher_params: Dict with hasher parameters
    
    Returns:
        List of (user_id, score) tuples for this chunk
    """
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.models_serialization import template_from_secure_dict
        from src.template_creation import CancelableHasher
        from src.matching import FingerprintMatcher
        
        # Deserialize probe
        probe_template = template_from_secure_dict(probe_secure_dict)
        
        # Deserialize gallery chunk
        gallery_templates = [
            template_from_secure_dict(tpl_dict) 
            for tpl_dict in gallery_chunk_secure_dicts.values()
        ]
        
        # Create hasher
        hasher = CancelableHasher(
            feature_dim=hasher_params['feature_dim'],
            projection_dim=hasher_params['projection_dim'],
            key=hasher_params['key'],
            hash_count=hasher_params['hash_count']
        )
        
        # Create matcher for this chunk
        matcher = FingerprintMatcher(gallery_templates, hasher)
        
        # Match probe against this chunk
        results = matcher.identify(
            probe_template, 
            top_k=len(gallery_templates),  # Get all matches from this chunk
            use_geometric_reranking=True
        )
        
        return results  # List of (user_id, score)
        
    except Exception as e:
        print(f"[ERROR worker_match_chunk] {str(e)}")
        return []


def worker_enroll_single_user(
    user_id: str,
    image_paths: List[str],
    settings: dict
) -> dict:
    """
    Worker function to enroll a single user (for parallel processing).
    
    Args:
        user_id: User identifier
        image_paths: List of image file paths for this user
        settings: Dict with 'quality_threshold' and 'fusion' settings
    
    Returns:
        {
            'success': bool,
            'user_id': str,
            'template_secure': dict (secure serialization),
            'quality': float,
            'num_images': int,
            'error': Optional[str]
        }
    """
    try:
        # Import supermatcher INSIDE worker (avoid pickling issues)
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.supermatcher_v1_0 import enroll
        from src.models import FusionSettings
        from src.models_serialization import template_to_secure_dict
        
        # Convert string paths to Path objects
        paths = [Path(p) for p in image_paths]
        
        # Create fusion settings
        fusion_dict = settings.get('fusion', {})
        fusion = FusionSettings(
            enabled=fusion_dict.get('enabled', True),
            distance=fusion_dict.get('distance', 12.0),
            angle_deg=fusion_dict.get('angle_deg', 15.0),
            min_consensus=fusion_dict.get('min_consensus', 0.5),
            keep_raw=fusion_dict.get('keep_raw', False),
            mode=fusion_dict.get('mode', 'optimal')
        )
        
        # Enroll user
        templates = enroll(
            image_paths=paths,
            identifier=user_id,
            fusion_settings=fusion,
            quality_threshold=settings.get('quality_threshold', 0.30),
            verbose=False
        )
        
        if not templates:
            return {
                'success': False,
                'user_id': user_id,
                'template_secure': None,
                'quality': 0.0,
                'num_images': len(image_paths),
                'error': 'No valid templates extracted'
            }
        
        template = templates[0]
        
        return {
            'success': True,
            'user_id': user_id,
            'template_secure': template_to_secure_dict(template),
            'quality': float(template.quality),
            'num_images': len(image_paths),
            'error': None
        }
    
    except Exception as e:
        return {
            'success': False,
            'user_id': user_id,
            'template_pickle': None,
            'quality': 0.0,
            'num_images': len(image_paths) if image_paths else 0,
            'error': str(e)
        }


def worker_enroll(
    image_paths: List[str],
    user_id: str,
    settings: dict
) -> dict:
    """
    Worker function for enrollment (runs in ProcessPool).
    
    Args:
        image_paths: List of image file paths
        user_id: User identifier
        settings: Dict with 'quality_threshold' and 'fusion' settings
    
    Returns:
        {
            'success': bool,
            'template_secure': dict (secure serialization),
            'quality': float,
            'num_images': int,
            'error': Optional[str]
        }
    """
    try:
        # Import supermatcher INSIDE worker (avoid pickling issues)
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.supermatcher_v1_0 import enroll
        from src.models import FusionSettings
        from src.models_serialization import template_to_secure_dict
        
        # Convert paths to Path objects
        paths = [Path(p) for p in image_paths]
        
        # Create fusion settings from dict
        fusion_dict = settings.get('fusion', {})
        
        # Map frontend params to FusionSettings params
        fusion = FusionSettings(
            enabled=fusion_dict.get('enabled', True),
            distance=fusion_dict.get('distance', 12.0),
            angle_deg=fusion_dict.get('angle_deg', 15.0),
            min_consensus=fusion_dict.get('min_consensus', 0.5),
            keep_raw=fusion_dict.get('keep_raw', False),
            mode=fusion_dict.get('mode', 'optimal')
        )
        
        # Enroll
        templates = enroll(
            image_paths=paths,
            identifier=user_id,
            fusion_settings=fusion,
            quality_threshold=settings.get('quality_threshold', 0.30),
            verbose=False
        )
        
        if not templates:
            return {
                'success': False,
                'template_secure': None,
                'quality': 0.0,
                'num_images': 0,
                'error': 'No valid templates extracted'
            }
        
        # Get master template
        template = templates[0]
        
        return {
            'success': True,
            'template_secure': template_to_secure_dict(template),
            'quality': float(template.quality),
            'num_images': len(image_paths),
            'error': None
        }
    
    except Exception as e:
        return {
            'success': False,
            'template_secure': None,
            'quality': 0.0,
            'num_images': 0,
            'error': str(e)
        }


def worker_verify(
    probe_path: str,
    template_secure_dict: Dict[str, Any],
    threshold: float,
    settings: dict
) -> dict:
    """
    Worker function for 1:1 verification.
    
    Args:
        probe_path: Path to probe image
        template_secure_dict: Secure dict of enrolled FingerprintTemplate
        threshold: Verification threshold
        settings: Dict with matching settings
    
    Returns:
        {
            'success': bool,
            'match': bool,
            'score': float,
            'num_matched': int,
            'error': Optional[str]
        }
    """
    try:
        # Import supermatcher INSIDE worker
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.supermatcher_v1_0 import FingerprintPipeline
        from src.template_creation import CancelableHasher
        from src.config import HASH_FEATURE_DIM, HASH_PROJECTION_DIM, HASH_KEY, HASH_COUNT
        from src.models_serialization import template_from_secure_dict
        
        # Deserialize enrolled template
        enrolled_template = template_from_secure_dict(template_secure_dict)
        
        # Create pipeline and process probe image (eliminates code duplication!)
        hasher = CancelableHasher(
            feature_dim=HASH_FEATURE_DIM,
            projection_dim=HASH_PROJECTION_DIM,
            key=HASH_KEY,
            hash_count=HASH_COUNT
        )
        
        pipeline = FingerprintPipeline(
            hasher=hasher,
            include_level3=False
        )
        
        probe_template = pipeline.process(
            image_path=Path(probe_path),
            identifier="probe",
            verbose=False
        )
        
        # Get hash similarity
        hash_similarity = hasher.similarity(probe_template.protected, enrolled_template.protected)
        
        # Compute geometric score
        from src.matching import compute_geometric_minutiae_score
        from src.config import MATCH_DISTANCE_THRESHOLD, MATCH_ANGLE_THRESHOLD_RAD
        
        probe_min = probe_template.minutiae or []
        enrolled_min = enrolled_template.minutiae or []
        
        geometric_score = compute_geometric_minutiae_score(
            probe_min,
            enrolled_min,
            distance_threshold=MATCH_DISTANCE_THRESHOLD,
            angle_threshold=MATCH_ANGLE_THRESHOLD_RAD,
        )
        
        # Combined score: 60% hash + 40% geometric (matching config)
        combined_score = 0.60 * hash_similarity + 0.40 * geometric_score
        match = combined_score >= threshold
        
        # Approximate number of matched minutiae (from geometric score)
        # geometric_score is roughly (matched / max(n_probe, n_candidate))
        max_minutiae = max(len(probe_min), len(enrolled_min)) if (probe_min and enrolled_min) else 0
        num_matched = int(geometric_score * max_minutiae) if geometric_score > 0 and max_minutiae > 0 else 0
        
        # DEBUG: Print detailed info
        print(f"[DEBUG worker_verify] hash={hash_similarity:.4f}, geometric={geometric_score:.4f}, "
              f"combined={combined_score:.4f}, threshold={threshold:.2f}, match={match}, "
              f"num_matched={num_matched}/{max_minutiae}")
        
        return {
            'success': True,
            'match': match,
            'score': float(combined_score),
            'hash_score': float(hash_similarity),
            'geometric_score': float(geometric_score),
            'num_matched': num_matched,
            'error': None
        }
    
    except Exception as e:
        return {
            'success': False,
            'match': False,
            'score': 0.0,
            'num_matched': 0,
            'error': str(e)
        }


def worker_identify(
    probe_path: str,
    gallery_secure_dicts: Dict[str, Dict[str, Any]],
    threshold: float,
    settings: dict,
    top_k: int = 5
) -> dict:
    """
    Worker function for 1:N identification.
    
    Args:
        probe_path: Path to probe image
        gallery_secure_dicts: Dict mapping user_id -> secure dict of FingerprintTemplate
        threshold: Identification threshold
        settings: Dict with matching settings
        top_k: Number of top matches to return
    
    Returns:
        {
            'success': bool,
            'matches': List[{'user_id': str, 'score': float, 'num_matched': int}],
            'identified': bool,
            'best_match_id': Optional[str],
            'best_score': float,
            'error': Optional[str]
        }
    """
    try:
        # Import supermatcher INSIDE worker
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.supermatcher_v1_0 import FingerprintPipeline
        from src.template_creation import CancelableHasher
        from src.matching import FingerprintMatcher
        from src.config import HASH_FEATURE_DIM, HASH_PROJECTION_DIM, HASH_KEY, HASH_COUNT
        from src.models_serialization import template_from_secure_dict, template_to_secure_dict
        
        # Deserialize all gallery templates
        gallery_templates = []
        for user_id, template_secure_dict in gallery_secure_dicts.items():
            template = template_from_secure_dict(template_secure_dict)
            gallery_templates.append(template)
        
        if not gallery_templates:
            return {
                'success': True,
                'matches': [],
                'identified': False,
                'best_match_id': None,
                'best_score': 0.0,
                'error': 'No templates in gallery'
            }
        
        # Create pipeline and process probe image (eliminates code duplication!)
        hasher = CancelableHasher(
            feature_dim=HASH_FEATURE_DIM,
            projection_dim=HASH_PROJECTION_DIM,
            key=HASH_KEY,
            hash_count=HASH_COUNT
        )
        
        pipeline = FingerprintPipeline(
            hasher=hasher,
            include_level3=False
        )
        
        probe_template = pipeline.process(
            image_path=Path(probe_path),
            identifier="probe",
            verbose=False
        )
        
        # OPTIMIZATION: Use multiprocessing for large galleries (≥20 users)
        gallery_size = len(gallery_templates)
        USE_PARALLEL = gallery_size >= 20
        
        if USE_PARALLEL:
            print(f"[DEBUG worker_identify] Large gallery ({gallery_size} users), using PARALLEL matching")
            
            # Prepare for parallel processing
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing
            
            # Serialize probe template once (secure dict)
            probe_secure_dict = template_to_secure_dict(probe_template)
            
            # Prepare hasher params
            hasher_params = {
                'feature_dim': HASH_FEATURE_DIM,
                'projection_dim': HASH_PROJECTION_DIM,
                'key': HASH_KEY,
                'hash_count': HASH_COUNT
            }
            
            # Split gallery into chunks (1 chunk per CPU core)
            num_workers = min(multiprocessing.cpu_count(), gallery_size)
            chunk_size = max(1, gallery_size // num_workers)
            
            gallery_items = list(gallery_secure_dicts.items())
            chunks = [
                dict(gallery_items[i:i + chunk_size])
                for i in range(0, len(gallery_items), chunk_size)
            ]
            
            print(f"[DEBUG] Splitting {gallery_size} templates into {len(chunks)} chunks, using {num_workers} workers")
            
            # Process chunks in parallel
            all_results = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(worker_match_probe_against_gallery_chunk, probe_secure_dict, chunk, hasher_params)
                    for chunk in chunks
                ]
                
                for future in as_completed(futures):
                    try:
                        chunk_results = future.result()
                        all_results.extend(chunk_results)
                    except Exception as e:
                        print(f"[ERROR identify_chunk] {str(e)}")
            
            # Sort all results by score (descending)
            all_results.sort(key=lambda x: x[1], reverse=True)
            
            # Take top_k
            results = all_results[:top_k]
            
        else:
            # SEQUENTIAL: For small galleries, use standard FingerprintMatcher
            print(f"[DEBUG worker_identify] Small gallery ({gallery_size} users), using SEQUENTIAL matching")
            
            matcher = FingerprintMatcher(gallery_templates, hasher)
            results = matcher.identify(probe_template, top_k=top_k, use_geometric_reranking=True)
        
        # Format results
        matches = [
            {
                'user_id': user_id,
                'score': float(score),
                'num_matched': 1 if score >= threshold else 0
            }
            for user_id, score in results
        ]
        
        identified = len(matches) > 0 and matches[0]['score'] >= threshold
        
        return {
            'success': True,
            'matches': matches,
            'identified': identified,
            'best_match_id': matches[0]['user_id'] if matches else None,
            'best_score': matches[0]['score'] if matches else 0.0,
            'error': None
        }
    
    except Exception as e:
        return {
            'success': False,
            'matches': [],
            'identified': False,
            'best_match_id': None,
            'best_score': 0.0,
            'error': str(e)
        }


def worker_load_folder(
    folder_path: str,
    settings: dict,
    progress_callback: Optional[Callable] = None
) -> dict:
    """
    Worker function for batch folder loading.
    
    Expects folder structure:
        folder_path/
            user1/
                img1.png
                img2.png
            user2/
                img1.png
                img2.png
    
    Args:
        folder_path: Path to folder containing user subfolders
        settings: Dict with enrollment settings
        progress_callback: Optional callback(user_id, status)
    
    Returns:
        {
            'success': bool,
            'enrolled': List[{'user_id': str, 'template_secure': dict, 'quality': float}],
            'failed': List[{'user_id': str, 'error': str}],
            'total': int,
            'error': Optional[str]
        }
    """
    try:
        # Import supermatcher INSIDE worker
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.supermatcher_v1_0 import enroll
        from src.models import FusionSettings
        
        folder = Path(folder_path)
        if not folder.exists():
            return {
                'success': False,
                'enrolled': [],
                'failed': [],
                'total': 0,
                'error': f'Folder not found: {folder_path}'
            }
        
        # Check if folder has subfolders or direct image files
        has_subfolders = any(d.is_dir() for d in folder.iterdir())
        
        print(f"[DEBUG worker_load_folder] folder_path={folder_path}")
        print(f"[DEBUG worker_load_folder] has_subfolders={has_subfolders}")
        
        enrolled = []
        failed = []
        
        # Two modes: subfolders (user1/, user2/) or flat structure (101_1.tif, 101_2.tif)
        if has_subfolders:
            # Original mode: folder/user1/img1.png, folder/user2/img1.png
            user_folders = [d for d in folder.iterdir() if d.is_dir()]
            print(f"[DEBUG] Subfolder mode: {len(user_folders)} user folders found")
            user_images_map = {f.name: list(f.iterdir()) for f in user_folders}
        else:
            # Flat mode: group files by prefix (101_1.tif, 101_2.tif -> user 101)
            image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
            all_files = [
                f for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            
            print(f"[DEBUG] Flat mode: {len(all_files)} image files found")
            
            # Group files by user ID (extract prefix before underscore or digit break)
            from collections import defaultdict
            user_images_map = defaultdict(list)
            
            for file in all_files:
                # Extract user ID from filename (e.g., "101_1.tif" -> "101")
                filename = file.stem  # Remove extension
                
                # Try to extract user ID (digits before underscore or end)
                import re
                match = re.match(r'^(\d+)', filename)
                if match:
                    user_id = match.group(1)
                    user_images_map[user_id].append(file)
                else:
                    # Skip files that don't match pattern
                    print(f"[DEBUG] Skipping file with no user ID pattern: {file.name}")
            
            print(f"[DEBUG] Grouped into {len(user_images_map)} users: {list(user_images_map.keys())}")
        
        # Prepare arguments for parallel processing
        print(f"[DEBUG] Starting parallel enrollment for {len(user_images_map)} users...")
        
        # Use multiprocessing to process users in parallel
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        
        # Use all available CPUs (or max 8)
        max_workers = min(multiprocessing.cpu_count(), 8)
        print(f"[DEBUG] Using {max_workers} worker processes")
        
        enrolled = []
        failed = []
        
        # Create list of tasks: (user_id, image_paths_as_strings, settings)
        tasks = []
        for user_id, image_paths in user_images_map.items():
            # Convert Path objects to strings for pickling
            image_path_strings = [str(p) for p in image_paths]
            tasks.append((user_id, image_path_strings, settings))
        
        # Process users in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_user = {
                executor.submit(worker_enroll_single_user, user_id, img_paths, settings): user_id
                for user_id, img_paths, settings in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_user):
                user_id = future_to_user[future]
                try:
                    result = future.result()
                    
                    if result['success']:
                        enrolled.append({
                            'user_id': result['user_id'],
                            'template_secure': result['template_secure'],
                            'quality': result['quality'],
                            'num_images': result['num_images']
                        })
                        print(f"[DEBUG] ✓ User {result['user_id']} enrolled (quality={result['quality']:.3f})")
                    else:
                        failed.append({
                            'user_id': result['user_id'],
                            'error': result['error']
                        })
                        print(f"[DEBUG] ✗ User {result['user_id']} failed: {result['error']}")
                    
                    if progress_callback:
                        progress_callback(user_id, 'completed' if result['success'] else 'failed')
                        
                except Exception as e:
                    failed.append({
                        'user_id': user_id,
                        'error': str(e)
                    })
                    print(f"[DEBUG] ✗ User {user_id} exception: {str(e)}")
                    
                    if progress_callback:
                        progress_callback(user_id, 'failed')
        
        print(f"[DEBUG] Load folder complete: {len(enrolled)} enrolled, {len(failed)} failed")
        
        return {
            'success': True,
            'enrolled': enrolled,
            'failed': failed,
            'total': len(user_images_map),
            'error': None
        }
    
    except Exception as e:
        return {
            'success': False,
            'enrolled': [],
            'failed': [],
            'total': 0,
            'error': str(e)
        }


# Test function
if __name__ == "__main__":
    print("Testing biometric worker functions...")
    
    # Note: These tests require actual images and templates
    # For real testing, provide actual paths
    
    print("\n✓ Worker functions defined successfully")
    print("  - worker_enroll()")
    print("  - worker_verify()")
    print("  - worker_identify()")
    print("  - worker_load_folder()")
