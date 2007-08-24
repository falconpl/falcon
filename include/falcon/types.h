/*
   FALCON - The Falcon Programming Language.
   FILE: ht_types.h
   $Id: types.h,v 1.2 2006/12/05 15:28:47 gian Exp $

   Declaration of types used by Falcon language.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-05-15
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#ifndef HT_TYPES_H
#define HT_TYPES_H

#include <falcon/setup.h>

namespace Falcon
{

class EngineData;

typedef char *   cstring;

typedef unsigned char byte;
typedef unsigned char * bytearray;

typedef byte uint8;
typedef unsigned short int uint16;
typedef unsigned int uint32;

#ifdef _MSC_VER
typedef unsigned __int64 uint64;
#else
typedef unsigned long long int uint64;
#endif

typedef char int8;
typedef short int int16;
typedef int int32;

#ifdef _MSC_VER
typedef __int64 int64;
#else
typedef long long int int64;
#endif

typedef double numeric;
typedef void * voidp;

class VMachine;
class Module;

typedef DLL void ( CDECL *ext_func_t) ( VMachine *);
typedef DLL Module* ( CDECL  *ext_mod_init)( const EngineData &data );

}

#endif
/* end of ht_types.h */
