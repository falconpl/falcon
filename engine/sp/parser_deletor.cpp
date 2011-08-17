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

#include <falcon/sp/parser_deletor.h>
#include "private_types.h"

#include <falcon/expression.h>

namespace Falcon {

void list_deletor(void* data)
{
   List* expr = static_cast<List*>(data);
   List::iterator iter = expr->begin();
   while( iter != expr->end() )
   {
      delete *iter;
      ++iter;
   }
   delete expr;
}

void expr_deletor(void* data)
{
   Expression* expr = static_cast<Expression*>(data);
   delete expr;
}



void pair_list_deletor(void* data)
{
   PairList* expr = static_cast<PairList*>(data);
   PairList::iterator iter = expr->begin();
   while( iter != expr->end() )
   {
      delete iter->first;
      delete iter->second;
      ++iter;
   }
   delete expr;
}


void name_list_deletor(void* data)
{
   delete static_cast<NameList*>(data);
}

}

/* end of parser_deletor.cpp */
