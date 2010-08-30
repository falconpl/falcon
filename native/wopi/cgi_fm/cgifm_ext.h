/*
   FALCON - The Falcon Programming Language.
   FILE: cgifm_ext.h

   Standalone CGI module for Falcon WOPI - ext declarations
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 28 Feb 2010 17:53:04 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
*/

#ifndef _FALCON_CGIFM_EXT_H_
#define _FALCON_CGIFM_EXT_H_

namespace Falcon
{
class VMachine;

void CGIRequest_init( VMachine* vm );
void CGIReply_init( VMachine* vm );

}

#endif

/* end of cgifm_ext.h */
