/*
   FALCON - The Falcon Programming Language.
   FILE: $FILE$.h
   $Id: gentree.h,v 1.3 2007/04/02 00:24:00 jonnymind Exp $

   Generates a compiler-debug oriented representation of the input symtree.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab giu 5 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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
