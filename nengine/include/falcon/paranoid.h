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
   #define PARANOID(...)
#else
   #include <falcon/codeerror.h>
   #define PARANOID( err, x ) \
         { if( !(x) ) { \
            throw new CodeError( ErrorParam(e_paranoid, __LINE__) \
               .module(__FILE__) \
               .extra(err) ); \
         }}
#endif

#endif	/* _FALCON_PARANOID_H */

/* end of paranoid.h */
