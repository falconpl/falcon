/*
   FALCON - The Falcon Programming Language.
   FILE: gentree.h

   Generates a compiler-debug oriented representation of the input symtree.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab giu 5 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#ifndef FALCON_GENTREE_H
#define FALCON_GENTREE_H

#include <falcon/setup.h>
#include <falcon/generator.h>

namespace Falcon
{
class Statement;
class Value;
class StatementList;
class Expression;
class ArrayDecl;
class DictDecl;

class FALCON_DYN_CLASS GenTree: public Generator
{
   void generate( const Statement *cmp, const char *spec=0, bool sameline = false, int depth=0 );
   void gen_value( const Value *val );
   void gen_block( const StatementList &blk, int depth, const char *prefix=0 );
   void gen_expression( const Expression *exp );
   void gen_array( const ArrayDecl *exp );
   void gen_dict( const DictDecl *ad );

public:
   GenTree( Stream *out ):
      Generator( out )
   {}

   virtual void generate( const SourceTree *st );

};

}
#endif

/* end of gentree.h */
