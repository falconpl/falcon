/*
   FALCON - The Falcon Programming Language.
   FILE: loadmode.h

   Common enumeration to define how a module is imported.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 09 Aug 2011 00:43:47 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_LOADMODE_H_
#define _FALCON_LOADMODE_H_

namespace Falcon {

   typedef enum 
   {
      e_lm_load,
      e_lm_import_public,
      e_lm_import_private            
   }
   t_loadMode;
   
}

#endif	/* _FALCON_LOADMODE_H_ */

/* end of loadmode.h */
