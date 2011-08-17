/*
   FALCON - The Falcon Programming Language.
   FILE: verion.h

   WINDOWS RC Versioning helper
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2007-07-25


   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   RC versioning helper.
   This small header is used in RC files, or by C programs, create version strings
   from version numbers stored in #define directive as numeric values.
*/

#include <falcon/setup.h>

#define FALCON_MAKE_VERSION_NUMBER(v1,v2,v3,v4)    v1, v2, v3, v4

#define FALCON_MAKE_VERSION_STRING(v1,v2,v3)    STR(v1) "." STR(v2) "."  STR(v3)
#define FALCON_MAKE_VERSION_STRING_4(v1,v2,v3,v4)    STR(v1) "." STR(v2) "."  STR(v3) "." STR(v4)

#define FALCON_MAKE_VERSION_STRING_RC(v1,v2,v3,v4) STR(v1) ", " STR(v2) ", "  STR(v3) ", " STR(v4)

#ifndef VS_VERSION_INFO
   #define VS_VERSION_INFO 1
#endif
