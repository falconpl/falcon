/*
   FALCON - The Falcon Programming Language.
   FILE: genhasm.h

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab giu 5 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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

   typedef enum {
      l_value,
      p_value,
      v_value
   } t_valType;

   /** Generates a complex value.
      Complex values are divided in two classes; expressions and structure constructions.

      Structure construction are namely:
      - range generations
      - array generation
      - dictionary generation
      - symbol reference generation

      Expresions are themselves divided into two kind; the ones that returns a "volatile"
      value, called l-value, and the ones that refer to an assignable, persistent location,
      called x-value. Namely, x-values are divided into p-values (expressions
      terminating with a property accesor) or v-value ( expression terminating with a
      sequence accessor).

      Some l-values:
      \code
         (a+1)
         functionCall()
         (a + b * c)
         var[ accessor ]()
         obj.property()
         obj.property[5].otherProp[2] + 1
      \endcode

      Some p-values:
      \code
         obj.property
         ( #"indirect"() ).property
         obj.property++    // we still refer to obj.property
      \endcode

      Some v-values:
      \code
         var[ accessorGenerator() ]
         obj.property[ x ]
         --var[5]          // we still refer to var[5]
      \endcode

      This distintion must be known, as assignments to l-values should be taken with care;
      for now, we do that and we assign to the A register. Conversely, assignment and auto-expressions
      on x-values is common.

      Instead of analyzing the structure of the expression, we let the underlying calls to the
      expression generators to determine if they are unrolling an l-value or an x-value.

      The parameter x_value is initially false; it gets changed to true if the last expanded
      expression element is an accessor. Operators ++ and -- don't change the character of the
      value, and all the others reset the character to l-value.

      \param value The value to be generated.
      \param x_value At exit, returns characteristic of the value.
   */

   void gen_complex_value( const Value *value, t_valType &x_value );

   /** Non l-value sensible version of complex value generator.
      Some operations do not require to know if the value generated is an l-value expression or not.
      They may use this version instead.
      \param value The value to be generated.
   */
   void gen_complex_value( const Value *value ) { t_valType dummy; gen_complex_value( value, dummy ); }

   /** Generate an expression.
      Expressions are the most common complex value. See gen_complex_value for a description.
      \param expr The expression to be generated.
      \param x_value On exit, characteristic value type of the expression.
      \see gen_complex_value
   */
   void gen_expression( const Expression *expr, t_valType &x_value );

   /** Generate an expression without taking its l-value characteristics.
      \param expr The expression to be generated.
   */
   void gen_expression( const Expression *expr ) { t_valType dummy; gen_expression( expr, dummy ); }

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

   void gen_inc_prefix( const Value *target );
   void gen_dec_prefix( const Value *target );
   void gen_inc_postfix( const Value *target );
   void gen_dec_postfix( const Value *target );
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

   class LoopInfo
   {
   public:
      LoopInfo( int id, const Statement* l ):
         m_loop( l ),
         m_id( id ),
         m_isForLast( false )
         {}

      const Statement* m_loop;
      int m_id;
      bool m_isForLast;
   };

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
