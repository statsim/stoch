/**
 * Resolve the common dtype between two dtypes.
 * float32 wins over int32, float32 is default.
 */
export function commonDtype(a, b) {
  if (a === 'float32' || b === 'float32') return 'float32'
  if (a === 'int32' && b === 'int32') return 'int32'
  return 'float32'
}

/**
 * Check if a dtype is a floating point type.
 */
export function isFloatDtype(dtype) {
  return dtype === 'float32'
}

/**
 * Check if a dtype is an integer type.
 */
export function isIntDtype(dtype) {
  return dtype === 'int32' || dtype === 'bool'
}
