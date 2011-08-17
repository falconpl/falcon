/*
   FALCON - The Falcon Programming Language.
   FILE: gencode.h

   Generates a compiler-debug oriented representation of the input symtree.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab giu 5 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#ifndef FALCON_GENCODE_H
#define FALCON_GENCODE_H

#include <falcon/generator.h>
#include <falcon/syntree.h>
#include <falcon/genericlist.h>
#include <falcon/genericvector.h>
#include <falcon/stringstream.h>

namespace Falcon
{

class LineMap;

class FALCON_DYN_CLASS GenCode: public Generator
{
   typedef enum {
      e_parND,
      e_parNIL,
      e_parUND,
      e_parVAL,
      e_parVAR,
      e_parINT32,
      e_parSYM,
      e_parA,
      e_parB,
      e_parS1,
      e_parL1,
      e_parL2,
      e_parLBIND,
      e_parTRUE,
      e_parFALSE,
      e_parNTD32,
      e_parNTD64,
      e_parSTRID
   } t_paramType;


   class c_jmptag
   {
      /* Varpar and jmptag are never newed, so they don't need operator new */
      uint32 m_defs[4];
      List m_queries[4];
      GenericVector m_elifDefs;
		GenericVector m_elifQueries;

      uint32 m_tries;
      uint32 m_offset;
      bool m_bIsForLast;
      const StmtForin* m_ForinLoop;
      Stream *m_stream;

      uint32 addQuery( uint32 id, uint32 pos );
      void define( uint32 id  );

   public:
      c_jmptag( Stream *stream, uint32 offset = 0 );

      uint32 addQueryBegin( uint32 blocks=1 ) { return addQuery( 0, blocks ); }
      uint32 addQueryEnd( uint32 blocks=1 ) { return addQuery( 1, blocks ); }
      uint32 addQueryNext( uint32 blocks=1 ) { return addQuery( 2, blocks ); }
      uint32 addQueryPostEnd( uint32 blocks=1 ) { return addQuery( 3, blocks ); }
      void defineBegin() { define( 0 ); }
      void defineEnd() { define( 1 ); }
      void defineNext() { define( 2 ); }
      void definePostEnd() { define( 3 ); }

      uint32 addQueryIfElse( uint32 blocks=1) { return addQuery( 0, blocks ); }
      uint32 addQueryIfEnd( uint32 blocks=1 ) { return addQuery( 1, blocks ); }
      uint32 addQueryElif( uint32 elifID, uint32 blocks=1 );
      uint32 addQuerySwitchBlock( uint32 blockID, uint32 blocks=1 )
      {
         return addQueryElif( blockID, blocks );
      }

      void defineIfElse() { define( 0 ); }
      void defineIfEnd() { define( 1 ); }
      void defineElif( uint32 id );
      void defineSwitchCase( uint32 id )
      {
         defineElif( id );
      }

      void addTry( uint32 count = 1 ) { m_tries += count; }
      void removeTry( uint32 count = 1 ) { m_tries -= count; }
      uint32 tryCount() const { return m_tries; }
      bool isForIn() const { return m_ForinLoop != 0; }
      const StmtForin* ForinLoop() const { return m_ForinLoop; }
      void setForIn( const StmtForin* forin ) { m_ForinLoop = forin; }
      void setForLast( bool l ) { m_bIsForLast = l; }
      bool isForLast() const { return m_bIsForLast; }
   };

   class c_varpar {
   public:
      /* Varpar and jmptag are never newed, so they don't need operator new */
      t_paramType m_type;
      union {
         const Value *value;
         const VarDef *vd;
         const Symbol *sym;
         int32 immediate;
         int64 immediate64;
      } m_content;

      c_varpar():
         m_type( e_parND )
      {}

      c_varpar( t_paramType t ):
         m_type( t )
      {}

      explicit c_varpar( bool val ):
         m_type( val ? e_parTRUE : e_parFALSE )
      {}

      c_varpar( const Value *val ):
         m_type( e_parVAL )
      {
         if ( val->isNil() )
            m_type = e_parNIL;
         else if( val->isUnbound() )
            m_type = e_parUND;
         else
            m_content.value = val;
      }

      c_varpar( const VarDef *vd ):
         m_type( e_parVAR )
      {
         if ( vd->isNil() )
            m_type = e_parNIL;
         else
            m_content.vd = vd;
      }

      c_varpar( const Symbol *sym ):
         m_type( e_parSYM )
      {
         m_content.sym = sym;
      }

      c_varpar( const int32 immediate ):
         m_type( e_parINT32 )
      {
         m_content.immediate = immediate;
      }

      c_varpar( const c_varpar &other ):
         m_type( other.m_type ),
         m_content( other.m_content )
      {}

      void generate( GenCode *owner ) const;

   };

   friend class c_varpar;

   c_varpar c_param_fixed( uint32 num ) {
      c_varpar ret( e_parNTD32 );
      ret.m_content.immediate = num;
      return ret;
   }

   c_varpar c_param_str( uint32 num ) {
      c_varpar ret( e_parSTRID );
      ret.m_content.immediate = num;
      return ret;
   }

   void gen_pcode( byte pcode )
   {
      gen_pcode( pcode, e_parND, e_parND, e_parND );
   }

   void gen_pcode( byte pcode, const c_varpar &first )
   {
      gen_pcode( pcode, first, e_parND, e_parND );
   }

   void gen_pcode( byte pcode, const c_varpar &first, const c_varpar &second )
   {
      gen_pcode( pcode, first, second, e_parND );
   }

   void gen_pcode( byte pcode, const c_varpar &first, const c_varpar &second, const c_varpar &third );
   byte gen_pdef( const c_varpar &elem );
   void gen_var( const VarDef &def );

   /* Pushes a label with a certain displacement.
      This function marks the position of a label in the next N int32 blocks, where
      N is the parameter. Usually, the jump or similar instruction has the jump
      target in the first int32 parameter, so the default for the param is 1.

      In case the position of the label target cannot be determined in advance,
      output operations must be perfomed manually, but at the moment all the functions
      with jump targets have only one or more int32 parmeters.

      \param displacement the distance in int32 block from current write position where label
             value must be written.
   */

   /** Writes a previously recorede label.
      This pops the label definition from the stack, writes the current file position
      and moves the file pointer back to where it was.

      This is the only operation causing write pointer to move (except for write).
   */
   void pop_label();

   void gen_function( const StmtFunction *func );
   void gen_block( const StatementList *slist );
   void gen_statement( const Statement *stmt );
   void gen_condition( const Value *stmt, int mode=-1 );
   void gen_value( const Value *stmt );

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
   void gen_store_to_deep( byte type, const Value *source, const Value *first, const Value *second );

   void gen_inc_prefix( const Value *target );
   void gen_dec_prefix( const Value *target );
   void gen_inc_postfix( const Value *target );
   void gen_dec_postfix( const Value *target );
   void gen_autoassign( byte opcode, const Value *target, const Value *source );
   void gen_store_to_deep_A( byte type, const Value *first, const Value *second );
   void gen_store_to_deep_reg( byte type, const Value *first, const Value *second, t_paramType reg );
   void gen_load_from_deep( byte type, const Value *first, const Value *second );
   void gen_load_from_A( const Value *target );
   void gen_load_from_reg( const Value *target, t_paramType reg );
   int gen_refArray( const ArrayDecl *tarr, bool bGenArray );
   void gen_operand( const Value *stmt );

   /** Generates a function call from an expression. */
   void gen_funcall( const Expression *call, bool fork=false );
   List m_labels;
   List m_labels_loop;

   /** Current context of functions, needed i.e. for states. */
   List m_functions;

   uint32 m_pc;
   StringStream *m_outTemp;
   Module *m_module;

public:
   GenCode( Module *mod );
   virtual ~GenCode();

   virtual void generate( const SourceTree *st );
};

}
#endif

/* end of gencode.h */
