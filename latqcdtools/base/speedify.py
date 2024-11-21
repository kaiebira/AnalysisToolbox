import os
import numpy as np
from numba import njit
from numba.typed import List
from typing import Any, Callable
import multiprocessing as mp
import latqcdtools.base.logger as logger

# Import JAX if available
try:
    import jax.numpy as jnp
    from jax import vmap, jit
    HAS_JAX = True
    # Configure JAX
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
except ImportError:
    HAS_JAX = False
    logger.debug('JAX not found - GPU acceleration disabled')

# Global configuration flags
COMPILENUMBA = False
COMPILEJAX = False
USE_MULTIPROCESSING = False
MAXTHREADS = os.cpu_count()
DEFAULTTHREADS = max(1, MAXTHREADS - 2)
NPROC = DEFAULTTHREADS

def numbaON():
    global COMPILENUMBA
    COMPILENUMBA = True
    logger.info('Using numba to speed things up.')

def numbaOFF():
    global COMPILENUMBA
    COMPILENUMBA = False
    logger.info('No longer using numba.')

def jaxON():
    if not HAS_JAX:
        logger.warn('JAX not installed - cannot enable GPU acceleration')
        return
    global COMPILEJAX
    COMPILEJAX = True
    logger.info('Using JAX for GPU acceleration.')

def jaxOFF():
    global COMPILEJAX
    COMPILEJAX = False
    logger.info('No longer using JAX.')

def multiprocessingON(nproc=None):
    global USE_MULTIPROCESSING, NPROC
    USE_MULTIPROCESSING = True
    if nproc is not None:
        NPROC = min(max(1, nproc), MAXTHREADS)
    logger.info(f'Using multiprocessing with {NPROC} processes.')

def multiprocessingOFF():
    global USE_MULTIPROCESSING
    USE_MULTIPROCESSING = False
    logger.info('No longer using multiprocessing.')

def handle_jax_array(x: Any) -> Any:
    """Safely convert arrays to JAX arrays"""
    if HAS_JAX and isinstance(x, (np.ndarray, list, tuple)):
        return jnp.asarray(x)
    return x

def handle_jax_args(args):
    """Convert arguments to JAX arrays where appropriate"""
    return tuple(handle_jax_array(arg) for arg in args)

def handle_jax_kwargs(kwargs):
    """Convert keyword arguments to JAX arrays where appropriate"""
    return {k: handle_jax_array(v) for k, v in kwargs.items()}

def _mp_worker(args):
    """Worker function for multiprocessing"""
    func, x, extra_args = args
    return func(x, *extra_args)

def compile(func: Callable) -> Callable:
    """
    Smart compiler that chooses between JAX, numba, or no compilation.
    """
    global COMPILEJAX, COMPILENUMBA
    
    if COMPILEJAX and HAS_JAX:
        logger.info(f'JAX-compiling {func.__name__}')
        try:
            @jit
            def jax_func(*args, **kwargs):
                # Convert inputs to JAX arrays when possible
                jax_args = handle_jax_args(args)
                jax_kwargs = handle_jax_kwargs(kwargs)
                
                # Run the function with JAX arrays
                result = func(*jax_args, **jax_kwargs)
                
                # Return as is - let the caller handle conversion if needed
                return result
                
            def wrapped_func(*args, **kwargs):
                try:
                    return jax_func(*args, **kwargs)
                except Exception as e:
                    logger.warn(f"JAX execution failed: {str(e)}. Using original function.")
                    return func(*args, **kwargs)
                    
            return wrapped_func
            
        except Exception as e:
            logger.warn(f'JAX compilation failed: {str(e)}. Falling back to numba.')
            
    if COMPILENUMBA:
        logger.info(f'Numba-compiling {func.__name__}')
        return njit(func)
        
    return func

def parallel_function_eval(function: Callable, input_array: Any, args=()) -> np.ndarray:
    """
    Parallelize function over input array using JAX or CPU multiprocessing.
    """
    # Try JAX first if enabled
    if COMPILEJAX and HAS_JAX:
        try:
            # Convert inputs to JAX arrays
            jax_input = jnp.asarray(input_array)
            jax_args = handle_jax_args(args)
            
            # Define the mapped function
            def mapped_func(x):
                return function(x, *jax_args)
            
            # Use JAX's vmap
            # Note: vmap is a vectorizing transformation, not a parallelization transformation. This batches the function over an array of inputs.
            batched = vmap(mapped_func)
            result = batched(jax_input)
            
            # Convert back to numpy if needed
            return np.array(result)
            
        except Exception as e:
            logger.warn(f'JAX parallelization failed: {str(e)}. Trying multiprocessing.')
    
    # Try multiprocessing if enabled
    if USE_MULTIPROCESSING:
        try:
            with mp.Pool(NPROC) as pool:
                mp_args = [(function, x, args) for x in input_array]
                results = pool.map(_mp_worker, mp_args)
            return np.array(results)
        except Exception as e:
            logger.warn(f'Multiprocessing failed: {str(e)}. Using sequential computation.')
    
    # Fallback to sequential computation
    return np.array([function(x, *args) for x in input_array])

def parallel_reduce(function: Callable, input_array: Any, args=()) -> Any:
    """
    Parallel reduction using JAX or CPU multiprocessing.
    """
    # Try JAX first if enabled
    if COMPILEJAX and HAS_JAX:
        try:
            # Convert inputs to JAX arrays
            jax_input = jnp.asarray(input_array)
            jax_args = handle_jax_args(args)
            
            # Define the mapped function
            def mapped_func(x):
                return function(x, *jax_args)
            
            # Use JAX's vmap and sum
            batched = vmap(mapped_func)
            result = jnp.sum(batched(jax_input))
            
            # Convert back to numpy if needed
            return np.array(result)
            
        except Exception as e:
            logger.warn(f'JAX reduction failed: {str(e)}. Trying multiprocessing.')
    
    # Try multiprocessing if enabled
    if USE_MULTIPROCESSING:
        try:
            with mp.Pool(NPROC) as pool:
                mp_args = [(function, x, args) for x in input_array]
                results = pool.map(_mp_worker, mp_args)
            return np.sum(results)
        except Exception as e:
            logger.warn(f'Multiprocessing reduction failed: {str(e)}. Using sequential reduction.')
    
    # Fallback to sequential reduction
    results = parallel_function_eval(function, input_array, args=args)
    return np.sum(results)

def numbaList(inList):
    """Convert list to numba List type if numba is enabled"""
    global COMPILENUMBA
    if COMPILENUMBA:
        nList = List()
        [nList.append(x) for x in inList]
        return nList 
    return inList
