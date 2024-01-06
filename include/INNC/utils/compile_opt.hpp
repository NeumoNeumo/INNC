#pragma once

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define __LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define __UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define __LIKELY(expr) (expr)
#define __UNLIKELY(expr) (expr)
#endif
