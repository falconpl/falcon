/*
   FALCON - The Falcon Programming Language.
   FILE: parser_deletor.h

   Falcon source parser -- Deletors used in parsing expressions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 17:34:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_SP_PARSER_DELETOR_H_
#define _FALCON_SP_PARSER_DELETOR_H_

#include <falcon/setup.h>

namespace Falcon {

void list_deletor(void* data);
void expr_deletor(void* data);
void pair_list_deletor(void* data);
void name_list_deletor(void* data);

}

#endif

/* end of parser_deletor.h */
