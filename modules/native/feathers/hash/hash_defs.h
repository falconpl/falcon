#ifndef _HASH_DEFS_H_
#define _HASH_DEFS_H_

#include <falcon/types.h>
#include <falcon/config.h>

#ifdef _MSC_VER
#  define UI64LIT(x) (x ## ui64)
#else
#  define UI64LIT(x) (x ## ULL)
#endif

typedef Falcon::uint64 word64;
typedef Falcon::uint32 word32;
typedef Falcon::byte byte;

#if (!defined(FALCON_LITTLE_ENDIAN)) || FALCON_LITTLE_ENDIAN == 0
#  define BIG_ENDIAN
#endif

#endif
