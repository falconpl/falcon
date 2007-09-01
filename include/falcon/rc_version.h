/*
   FALCON - The Falcon Programming Language.
   FILE: verion.h
   $Id: version.h,v 1.1 2007/07/25 12:23:08 jonnymind Exp $

   WINDOWS RC Versioning helper
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2007-07-25

   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file 
   RC versioning helper.
   This small header is used in RC files, or by C programs, create version strings
   from version numbers stored in #define directive as numeric values.
*/

#define _STR(x) #x
#define STR(x) _STR(x)

#define FALCON_MAKE_VERSION_NUMBER(v1,v2,v3,v4)    v1, v2, v3, v4

#define FALCON_MAKE_VERSION_STRING(v1,v2,v3)    STR(v1) "." STR(v2) "."  STR(v3)
#define FALCON_MAKE_VERSION_STRING_4(v1,v2,v3,v4)    STR(v1) "." STR(v2) "."  STR(v3) "." STR(v4)

#define FALCON_MAKE_VERSION_STRING_RC(v1,v2,v3,v4) STR(v1) ", " STR(v2) ", "  STR(v3) ", " STR(v4)

#ifndef VS_VERSION_INFO
   #define VS_VERSION_INFO 1
#endif
