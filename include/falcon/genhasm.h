/*
   FALCON - The Falcon Programming Language.
   FILE: genhasm.h
   $Id: genhasm.h,v 1.7 2007/07/10 20:40:18 jonnymind Exp $

   Short description
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

#ifndef FALCON_GENHSAM_H
#define FALCON_GENHSAM_H

#include <falcon/generator.h>
#include <falcon/compiler.h>
#include <falcon/genericlist.h>

namespace Falcon
{


class FALCON_DYN_CLASS GenHAsm: public Generator
{
   void gen_depTable( const Module *mod );
   void gen_stringTable( const Module *mod );
   void gen_symbolTable( const Module *mod );
   void gen_function( const StmtFunction *func );
   void gen_class( const StmtClass *cls );
   void gen_propdef( const VarDef &def );
   void gen_block( const StatementList *slist );
   void gen_statement( const Statement *stmt );
   void gen_condition( const Value *stmt, int mode=0 );
   void gen_value( const Value *stmt, const char *prefix = 0, const char *cpl_post = 0 );
   void gen_operand( const Value *stmt );
   void gen_complex_value( const Value *stmt, bool assign = false );
   void gen_expression( const Expression *stmt, bool assign = false );
   void gen_dict_decl( const DictDecl *stmt );
   void gen_array_decl( const ArrayDecl *stmt );
   void gen_range_decl( const RangeDecl *stmt );
   /** Geneare a push instruction based on a source value.
      Push is a kinda tricky instruction. Theoretically, if the value holds
      a reference taking expression, one can LDRF on the target and then push A,
      but PSHR is provided to do this in just one step. Hence the need of a specialized
      push generator that is a little smarter than the usual gen_value.
   */
   void gen_push( const Value *val );
   /** Generate a load instruction.
      LD is a kinda tricky instruction. Theoretically, if the source value holds
      a reference taking expression, one can LDRF on the target and then load A into the source
      but LDRF is provided to do this in just one step. Hence the need of a specialized
      load generator that is a little smarter.
   */
   void gen_load( const Value *target, const Value *source );
   //void gen_load( const Value *target, const char *source );
   void gen_store_to_deep( const char *type, const Value *source, const Value *first, const Value *second );

   void gen_inc_prefix( const Value *target, bool asExpr = false );
   void gen_dec_prefix( const Value *target, bool asExpr = false );
   void gen_inc_postfix( const Value *target, bool asExpr = false );
   void gen_dec_postfix( const Value *target, bool asExpr = false );
   void gen_autoassign( const char *op, const Value *target, const Value *source );
   void gen_store_to_deep_A( const char *type, const Value *first, const Value *second );
   void gen_store_to_deep_reg( const char *type, const Value *first, const Value *second, const char *reg );
   void gen_load_from_deep( const char *type, const Value *first, const Value *second );
   void gen_load_from_A( const Value *target );
   void gen_load_from_reg( const Value *target, const char *reg );
   int gen_refArray( const ArrayDecl *tarr, bool bGenArray );

   /** Writes a set of cases. */
   void dump_cases( int branch, const MapIterator &begin );

   /** Generates a function call from an expression. */
   void gen_funcall( const Expression *call, bool fork=false );

   int m_branch_id;
   int m_loop_id;
   int m_try_id;

   List m_branches;
   List m_loops;
   List m_trys;

   /** Current context of functions, needed i.e. for states. */
   List m_functions;

public:
   GenHAsm( Stream *out );
   virtual void generate( const SourceTree *st );
   void generatePrologue( const Module *module );
};

}
#endif
/* end of genhasm.h */
