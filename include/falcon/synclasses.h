/*
   FALCON - The Falcon Programming Language.
   FILE: synclasses.h

   Class holding all the Class reflector for syntactic tree elements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 29 Dec 2011 12:07:48 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_SYNCLASSES_H
#define FALCON_SYNCLASSES_H

#include <falcon/setup.h>
#include <falcon/derivedfrom.h>

#define FALCON_SYNCLASS_DECLARATOR_EX( name, type, extra ) \
   class Class##name: public DerivedFrom\
   {public:\
      Class##name(): DerivedFrom( m_cls_##type, "##name" ) {}   \
      virtual ~Class##name() {};\
      virtual void op_create( VMContext* ctx, int32 pcount ) const;\
      virtual TreeStep* createInstance() const; \
      extra\
   }
   
#define FALCON_SYNCLASS_DECLARATOR( name, type ) \
   FALCON_SYNCLASS_DECLARATOR_EX(name, type,)
   

namespace Falcon {

class Engine;
class VMContext;
class GarbageToken;

/** Class holding all the Class reflector for syntactic tree elements.
 
 */
class SynClasses
{
public:
   /** Creates the syntactic classes.
    \param classSynTree the ClassSynTree instance held by the engine.
    \param classStatement the ClassStatement instance held by the engine.
    \param classExpr the ClassExpr instance held by the engine.
    
    */
   SynClasses( Class* classSynTree, Class* classStatement, Class* classExpr ): 
      m_cls_st( classSynTree ),
      m_cls_stmt( classStatement ),
      m_cls_expr( classExpr )
   {}
      
   ~SynClasses();
   
   /** Subscribes all the syntactic classes to the engine.
    \param engine The engine where to subscribe the classes.
    
    This method adds all the Syntactic classes to the engine as registered
    classes.
    */
   void subscribe( Engine* engine );
   
   static GarbageToken* collect( Class*, TreeStep*, int line );
   
   static void varExprInsert( VMContext* ctx, int pcount, TreeStep* step );   
   static void naryExprSet( VMContext* ctx, int pcount, TreeStep* step, int32 size );

   static inline void unaryExprSet( VMContext* ctx, int pcount, TreeStep* step ) {
      naryExprSet( ctx, pcount, step, 1 );
   }
   
   static inline void binaryExprSet( VMContext* ctx, int pcount, TreeStep* step ) {
      naryExprSet( ctx, pcount, step, 2 );
   }
   
   static inline void ternaryExprSet( VMContext* ctx, int pcount, TreeStep* step ) {
      naryExprSet( ctx, pcount, step, 3 );
   }
   
   static inline void zeroaryExprSet( VMContext*, int, TreeStep* ) {
      // no need to do anything
   }
   
   //======================================================================
   // Ecpression classes
   //
   FALCON_SYNCLASS_DECLARATOR(GenArray, expr) m_expr_genarray;
   FALCON_SYNCLASS_DECLARATOR(Assign, expr) m_expr_assign;
   FALCON_SYNCLASS_DECLARATOR(BNot, expr) m_expr_bnot;
   FALCON_SYNCLASS_DECLARATOR(Call, expr) m_expr_call;
   // compare
   FALCON_SYNCLASS_DECLARATOR(LT, expr) m_expr_lt;
   FALCON_SYNCLASS_DECLARATOR(LE, expr) m_expr_le;
   FALCON_SYNCLASS_DECLARATOR(GT, expr) m_expr_gt;
   FALCON_SYNCLASS_DECLARATOR(GE, expr) m_expr_ge;
   FALCON_SYNCLASS_DECLARATOR(EQ, expr) m_expr_eq;
   FALCON_SYNCLASS_DECLARATOR(NE, expr) m_expr_ne;
   //
   FALCON_SYNCLASS_DECLARATOR(GenDict, expr) m_expr_gendict;
   FALCON_SYNCLASS_DECLARATOR(DotAccess, expr) m_expr_dot;
   FALCON_SYNCLASS_DECLARATOR(EEQ, expr) m_expr_eeq;
   FALCON_SYNCLASS_DECLARATOR(IIF, expr) m_expr_iif;
   // inc-dec
   FALCON_SYNCLASS_DECLARATOR(PreInc, expr) m_expr_preinc;
   FALCON_SYNCLASS_DECLARATOR(PreDec, expr) m_expr_predec;
   FALCON_SYNCLASS_DECLARATOR(PostInc, expr) m_expr_postinc;
   FALCON_SYNCLASS_DECLARATOR(PostDec, expr) m_expr_postdec;
   //
   FALCON_SYNCLASS_DECLARATOR(IndexAccess, expr) m_expr_index;
   // Logic
   FALCON_SYNCLASS_DECLARATOR(Not, expr) m_expr_not;
   FALCON_SYNCLASS_DECLARATOR(And, expr) m_expr_and;
   FALCON_SYNCLASS_DECLARATOR(Or, expr) m_expr_or;
   // Math
   FALCON_SYNCLASS_DECLARATOR(Plus, expr) m_expr_plus;
   FALCON_SYNCLASS_DECLARATOR(Minus, expr) m_expr_minus;
   FALCON_SYNCLASS_DECLARATOR(Times, expr) m_expr_times;
   FALCON_SYNCLASS_DECLARATOR(Div, expr) m_expr_div;
   FALCON_SYNCLASS_DECLARATOR(Mod, expr) m_expr_mod;
   FALCON_SYNCLASS_DECLARATOR(Pow, expr) m_expr_pow;
   FALCON_SYNCLASS_DECLARATOR(RShift, expr) m_expr_rshift;
   FALCON_SYNCLASS_DECLARATOR(LShift, expr) m_expr_lshift;
   FALCON_SYNCLASS_DECLARATOR(BAnd, expr) m_expr_band;
   FALCON_SYNCLASS_DECLARATOR(BOr, expr) m_expr_bor;
   FALCON_SYNCLASS_DECLARATOR(BXor, expr) m_expr_bxor;
   // Auto-math
   FALCON_SYNCLASS_DECLARATOR(AutoPlus, expr) m_expr_aplus;
   FALCON_SYNCLASS_DECLARATOR(AutoMinus, expr) m_expr_aminus;
   FALCON_SYNCLASS_DECLARATOR(AutoTimes, expr) m_expr_atimes;
   FALCON_SYNCLASS_DECLARATOR(AutoDiv, expr) m_expr_adiv;
   FALCON_SYNCLASS_DECLARATOR(AutoMod, expr) m_expr_amod;
   FALCON_SYNCLASS_DECLARATOR(AutoPow, expr) m_expr_apow;
   FALCON_SYNCLASS_DECLARATOR(AutoLShift, expr) m_expr_alshift;
   FALCON_SYNCLASS_DECLARATOR(AutoRShigt, expr) m_expr_arshift;
   // 
   FALCON_SYNCLASS_DECLARATOR(MUnpack, expr) m_expr_munpack;
   FALCON_SYNCLASS_DECLARATOR(Neg, expr) m_expr_neg;
   // OOB
   FALCON_SYNCLASS_DECLARATOR(Oob, expr) m_expr_oob;
   FALCON_SYNCLASS_DECLARATOR(DeOob, expr) m_expr_deoob;
   FALCON_SYNCLASS_DECLARATOR(IsOob, expr) m_expr_isoob;
   FALCON_SYNCLASS_DECLARATOR(XorOob, expr) m_expr_xoroob;
   // 
   FALCON_SYNCLASS_DECLARATOR(GenProto, expr) m_expr_genproto;
   FALCON_SYNCLASS_DECLARATOR(GenRange, expr) m_expr_genrange;
   FALCON_SYNCLASS_DECLARATOR(GenRef, expr) m_expr_genref;
   FALCON_SYNCLASS_DECLARATOR(Self, expr) m_expr_self;
   FALCON_SYNCLASS_DECLARATOR(StarIndexAccess, expr) m_expr_starindex;
   FALCON_SYNCLASS_DECLARATOR(GenSym, expr) m_expr_sym;
   FALCON_SYNCLASS_DECLARATOR(Unpack, expr) m_expr_unpack;
   FALCON_SYNCLASS_DECLARATOR(Value, expr) m_expr_value;
   
   //======================================================================
   // Statement classes
   //
   FALCON_SYNCLASS_DECLARATOR(AutoExpr, stmt) m_stmt_autoexpr;
   FALCON_SYNCLASS_DECLARATOR(Break, stmt) m_stmt_break;
   FALCON_SYNCLASS_DECLARATOR(Breakpoint, stmt) m_stmt_breakpoint;
   FALCON_SYNCLASS_DECLARATOR(Continue, stmt) m_stmt_continue;
   FALCON_SYNCLASS_DECLARATOR(Cut, stmt) m_stmt_cut;
   FALCON_SYNCLASS_DECLARATOR(Doubt, stmt) m_stmt_doubt;
   FALCON_SYNCLASS_DECLARATOR(FastPrint, stmt) m_stmt_fastprint;
   FALCON_SYNCLASS_DECLARATOR(ForIn, stmt) m_stmt_forin;
   FALCON_SYNCLASS_DECLARATOR(ForTo, stmt) m_stmt_forto;
   FALCON_SYNCLASS_DECLARATOR(If, stmt) m_stmt_if;
   FALCON_SYNCLASS_DECLARATOR(Init, stmt) m_stmt_init;
   FALCON_SYNCLASS_DECLARATOR(Raise, stmt) m_stmt_raise;
   FALCON_SYNCLASS_DECLARATOR(Return, stmt) m_stmt_return;
   FALCON_SYNCLASS_DECLARATOR(Rule, stmt) m_stmt_rule;
   FALCON_SYNCLASS_DECLARATOR(Select, stmt) m_stmt_select;
   FALCON_SYNCLASS_DECLARATOR(Try, stmt) m_stmt_try;
   FALCON_SYNCLASS_DECLARATOR(While, stmt) m_stmt_while;
   
   //======================================================================
   // Syntree classes
   //
   FALCON_SYNCLASS_DECLARATOR(RuleSynTree, m_cls_st) m_st_rulest;
   

   //======================================================================
   // Basee classes
   //
   Class* m_cls_st;
   Class* m_cls_stmt;
   Class* m_cls_expr;
};

}

#define FALCON_DECLARE_SYN_CLASS( syntoken ) \
   static Class* syntoken = &Engine::instance()->synclasses()->m_##syntoken; \
   m_class = syntoken;

#endif	/* SYNCLASSES_H */

/* end of synclasses.h */
