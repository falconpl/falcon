
#ifdef FALCON_SYNCLASS_DECLARATOR_DECLARE
#undef FALCON_SYNCLASS_DECLARATOR_EX
#undef FALCON_SYNCLASS_DECLARATOR
#define FALCON_SYNCLASS_DECLARATOR_EX( variable, name, type, extra ) \
   class Class##name: public DerivedFrom\
   {public:\
      Class##name( Class* derfrom ): DerivedFrom( derfrom, #name ) {}   \
      virtual ~Class##name() {};\
      virtual void* createInstance() const;\
      virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;\
      virtual void restore( VMContext* ctx, DataReader*dr, void*& empty ) const; \
      extra\
   }; \
   Class##name * variable;
   
#define FALCON_SYNCLASS_DECLARATOR( variable, name, type ) \
   FALCON_SYNCLASS_DECLARATOR_EX( variable, name, type,)

#else
#ifdef FALCON_SYNCLASS_DECLARATOR_APPLY

#undef FALCON_SYNCLASS_DECLARATOR_EX
#undef FALCON_SYNCLASS_DECLARATOR
#define FALCON_SYNCLASS_DECLARATOR_EX( variable, name, type, extra ) \
   variable = new Class##name( m_cls_##type );
   
#define FALCON_SYNCLASS_DECLARATOR( variable, name, type ) \
   FALCON_SYNCLASS_DECLARATOR_EX( variable, name, type,)


#else
#ifdef FALCON_SYNCLASS_DECLARATOR_REGISTER

#undef FALCON_SYNCLASS_DECLARATOR_EX
#undef FALCON_SYNCLASS_DECLARATOR

#define FALCON_SYNCLASS_DECLARATOR_EX( variable, name, type, extra ) \
   engine->addMantra(variable);
   
#define FALCON_SYNCLASS_DECLARATOR( variable, name, type ) \
   FALCON_SYNCLASS_DECLARATOR_EX( variable, name, type,)


#else
#define FALCON_SYNCLASS_DECLARATOR_EX( variable, name, type, extra ) 
#define FALCON_SYNCLASS_DECLARATOR( variable, name, type ) 
#endif
#endif
#endif


//======================================================================
// Expression classes
//
FALCON_SYNCLASS_DECLARATOR(m_expr_genarray, GenArray, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_assign, Assign, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_bnot, BNot, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_call, Call, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_closure, GenClosure, expr)
// compare
FALCON_SYNCLASS_DECLARATOR(m_expr_lt, LT, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_le, LE, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_gt, GT, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_ge, GE, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_eq, EQ, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_ne, NE, expr)
//
FALCON_SYNCLASS_DECLARATOR(m_expr_gendict, GenDict, expr)
FALCON_SYNCLASS_DECLARATOR_EX(m_expr_dot, DotAccess, expr, \
      virtual void store( VMContext*, DataWriter* dw, void* instance ) const; \
      )
FALCON_SYNCLASS_DECLARATOR(m_expr_eeq, EEQ, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_in, In, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_notin, Notin, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_eval, Eval, expr)   
FALCON_SYNCLASS_DECLARATOR(m_expr_iif, IIF, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_lit, Lit, expr)
// inc-dec
FALCON_SYNCLASS_DECLARATOR(m_expr_preinc, PreInc, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_predec, PreDec, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_postinc, PostInc, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_postdec, PostDec, expr)
//
FALCON_SYNCLASS_DECLARATOR(m_expr_index, IndexAccess, expr)

FALCON_SYNCLASS_DECLARATOR_EX(m_expr_inherit, Inherit, expr, \
      virtual void store( VMContext*, DataWriter* dw, void* instance ) const; \
      virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;\
      virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const; )

FALCON_SYNCLASS_DECLARATOR(m_expr_parentship, Parentship, expr )

// Logic
FALCON_SYNCLASS_DECLARATOR(m_expr_not, Not, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_and, And, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_or, Or, expr)
// Math
FALCON_SYNCLASS_DECLARATOR(m_expr_plus, Plus, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_minus, Minus, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_times, Times, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_div, Div, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_mod, Mod, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_pow, Pow, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_rshift, RShift, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_lshift, LShift, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_band, BAnd, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_bor, BOr, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_bxor, BXor, expr)
// Auto-math
FALCON_SYNCLASS_DECLARATOR(m_expr_aplus, AutoPlus, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_aminus, AutoMinus, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_atimes, AutoTimes, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_adiv, AutoDiv, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_amod, AutoMod, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_apow, AutoPow, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_alshift, AutoLShift, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_arshift, AutoRShift, expr)

// Functional
FALCON_SYNCLASS_DECLARATOR(m_expr_compose, Compose, expr )
FALCON_SYNCLASS_DECLARATOR(m_expr_funcpower, FuncPower, expr )

// 
FALCON_SYNCLASS_DECLARATOR(m_expr_munpack, MUnpack, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_neg, Neg, expr)
// OOB
FALCON_SYNCLASS_DECLARATOR(m_expr_oob, Oob, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_deoob, DeOob, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_isoob, IsOob, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_xoroob, XorOob, expr)
// 
FALCON_SYNCLASS_DECLARATOR(m_expr_pseudocall, PseudoCall, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_genproto, GenProto, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_genrange, GenRange, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_genref, GenRef, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_self, Self, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_starindex, StarIndexAccess, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_unpack, Unpack, expr)
FALCON_SYNCLASS_DECLARATOR(m_expr_unquote, Unquote, expr)

FALCON_SYNCLASS_DECLARATOR_EX(m_expr_sym, GenSym, expr, \
      virtual void store( VMContext*, DataWriter* dw, void* instance ) const; \
      virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;\
      virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const; )

FALCON_SYNCLASS_DECLARATOR_EX(m_expr_value, Value, expr, \
      virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;\
      virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const; )

   
   
   
   /** The normal behavior is that to unflatten all via nth.
   */
   
//======================================================================
// Statement classes
//
FALCON_SYNCLASS_DECLARATOR(m_stmt_autoexpr, AutoExpr, stmt)
FALCON_SYNCLASS_DECLARATOR(m_stmt_break, Break, stmt)
FALCON_SYNCLASS_DECLARATOR(m_stmt_breakpoint, Breakpoint, stmt)
FALCON_SYNCLASS_DECLARATOR(m_stmt_continue, Continue, stmt)
FALCON_SYNCLASS_DECLARATOR(m_stmt_cut, Cut, stmt)
FALCON_SYNCLASS_DECLARATOR(m_stmt_doubt, Doubt, stmt)
FALCON_SYNCLASS_DECLARATOR(m_stmt_fastprint, FastPrint, stmt)
FALCON_SYNCLASS_DECLARATOR(m_stmt_forin, ForIn, stmt)
FALCON_SYNCLASS_DECLARATOR(m_stmt_forto, ForTo, stmt)
FALCON_SYNCLASS_DECLARATOR(m_stmt_if, If, stmt)
FALCON_SYNCLASS_DECLARATOR(m_stmt_raise, Raise, stmt)
FALCON_SYNCLASS_DECLARATOR(m_stmt_return, Return, stmt)
FALCON_SYNCLASS_DECLARATOR(m_stmt_rule, Rule, stmt)
FALCON_SYNCLASS_DECLARATOR(m_stmt_select, Select, stmt)
FALCON_SYNCLASS_DECLARATOR(m_stmt_switch, Switch, stmt)
FALCON_SYNCLASS_DECLARATOR(m_stmt_try, Try, stmt)
FALCON_SYNCLASS_DECLARATOR(m_stmt_while, While, stmt)

//======================================================================
// Syntree classes
//
FALCON_SYNCLASS_DECLARATOR(m_st_rulest, RuleSynTree, st)
