/*
   FALCON - The Falcon Programming Language.
   FILE: paranoid.h

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab mar 18 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PARANOID_H
#define	_FALCON_PARANOID_H

#ifdef NDEBUG
#ifndef FALCON_PARANOID
#define _FALCON_NO_PARANOID_CHECK_
#endif
#endif

#ifdef _FALCON_NO_PARANOID_CHECK_
   #define PARANOID(...)
#else
   #include <falcon/errors/codeerror.h>
   #define PARANOID( errdesc, x ) \
         { if( !(x) ) { \
            throw new CodeError( ErrorParam(e_paranoid, __LINE__) \
               .module(__FILE__) \
               .extra(errdesc) ); \
         }}
#endif

#endif	/* _FALCON_PARANOID_H */

/* end of paranoid.h */
