
#ifdef FALCON_SYNCLASS_DECLARATOR_DECLARE
#undef FALCON_SYNCLASS_DECLARATOR_EX
#undef FALCON_SYNCLASS_DECLARATOR
#define FALCON_SYNCLASS_DECLARATOR_EX( variable, name, type, extra ) \
   class Class##name: public Class##type\
   {public:\
      Class##name( Class* derfrom ): Class##type( #name ) {setParent(derfrom);}   \
      virtual ~Class##name() {};\
      virtual void* createInstance() const;\
      virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;\
      virtual void restore( VMContext* ctx, DataReader*dr ) const; \
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
FALCON_SYNCLASS_DECLARATOR(m_expr_genarray, GenArray, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_assign, Assign, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_accumulator, Accumulator, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_bnot, BNot, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_call, Call, Expression)
FALCON_SYNCLASS_DECLARATOR_EX(m_expr_case, Case, Expression, \
         virtual void op_call(VMContext* ctx, int pcount, void* instance) const; \
         virtual void store( VMContext*, DataWriter* dw, void* instance ) const; \
         virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;\
         virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const; )

FALCON_SYNCLASS_DECLARATOR(m_expr_ep, EP, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_invoke, Invoke, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_stripol, StrIPol, Expression)
FALCON_SYNCLASS_DECLARATOR_EX(m_expr_closure, GenClosure, Expression, \
      virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;\
      virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const; )


// compare
FALCON_SYNCLASS_DECLARATOR(m_expr_lt, LT, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_le, LE, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_gt, GT, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_ge, GE, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_eq, EQ, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_ne, NE, Expression)
//
FALCON_SYNCLASS_DECLARATOR(m_expr_gendict, GenDict, Expression)
FALCON_SYNCLASS_DECLARATOR_EX(m_expr_dot, DotAccess, Expression, \
      virtual void store( VMContext*, DataWriter* dw, void* instance ) const; \
      )

FALCON_SYNCLASS_DECLARATOR(m_expr_iif, IIF, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_eeq, EEQ, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_in, In, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_notin, Notin, Expression)

FALCON_SYNCLASS_DECLARATOR_EX(m_expr_provides, Provides, Expression, \
         virtual void store( VMContext*, DataWriter* dw, void* instance ) const; \
         virtual void op_getProperty( VMContext* ctx, void* instance, const String& prop) const;\
         virtual void op_setProperty( VMContext* ctx, void* instance, const String& prop ) const;\
         virtual bool hasProperty( void*, const String& ) const; \
         )


FALCON_SYNCLASS_DECLARATOR_EX(m_expr_lit, Lit, Expression, \
         virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;\
         virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const; )

FALCON_SYNCLASS_DECLARATOR_EX(m_expr_tree, Tree, Expression, \
            void op_call(VMContext* ctx, int pcount, void* instance) const; \
            virtual void store( VMContext*, DataWriter* dw, void* instance ) const; \
            virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;\
            virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const; )

// inc-dec
FALCON_SYNCLASS_DECLARATOR(m_expr_preinc, PreInc, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_predec, PreDec, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_postinc, PostInc, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_postdec, PostDec, Expression)
//
FALCON_SYNCLASS_DECLARATOR(m_expr_index, IndexAccess, Expression)

FALCON_SYNCLASS_DECLARATOR_EX(m_expr_inherit, Inherit, Expression, \
      virtual void store( VMContext*, DataWriter* dw, void* instance ) const; \
      virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;\
      virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const; )

FALCON_SYNCLASS_DECLARATOR(m_expr_parentship, Parentship, Expression )

// Logic
FALCON_SYNCLASS_DECLARATOR(m_expr_not, Not, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_and, And, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_or, Or, Expression)
// Math
FALCON_SYNCLASS_DECLARATOR(m_expr_plus, Plus, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_minus, Minus, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_times, Times, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_div, Div, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_mod, Mod, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_pow, Pow, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_rshift, RShift, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_lshift, LShift, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_band, BAnd, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_bor, BOr, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_bxor, BXor, Expression)
// Auto-math
FALCON_SYNCLASS_DECLARATOR(m_expr_aplus, AutoPlus, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_aminus, AutoMinus, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_atimes, AutoTimes, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_adiv, AutoDiv, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_amod, AutoMod, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_apow, AutoPow, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_alshift, AutoLShift, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_arshift, AutoRShift, Expression)

// Functional
FALCON_SYNCLASS_DECLARATOR(m_expr_compose, Compose, Expression)

FALCON_SYNCLASS_DECLARATOR(m_expr_neg, Neg, Expression)
// OOB
FALCON_SYNCLASS_DECLARATOR(m_expr_oob, Oob, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_deoob, DeOob, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_isoob, IsOob, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_xoroob, XorOob, Expression)
// 
FALCON_SYNCLASS_DECLARATOR(m_expr_pseudocall, PseudoCall, Expression)
FALCON_SYNCLASS_DECLARATOR_EX(m_expr_genproto, GenProto, Expression, \
         virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;\
         virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const; )

FALCON_SYNCLASS_DECLARATOR(m_expr_genrange, GenRange, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_self, Self, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_fself, FSelf, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_init, Init, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_starindex, StarIndexAccess, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_unpack, Unpack, Expression )
FALCON_SYNCLASS_DECLARATOR(m_expr_munpack, MUnpack, Expression )


FALCON_SYNCLASS_DECLARATOR(m_expr_unquote, Unquote, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_evalret, EvalRet, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_evalretexec, EvalRetExec, Expression)
FALCON_SYNCLASS_DECLARATOR(m_expr_evalretdoubt, EvalRetDoubt, Expression)


FALCON_SYNCLASS_DECLARATOR_EX(m_expr_sym, GenSym, Expression, \
      virtual void store( VMContext*, DataWriter* dw, void* instance ) const; \
      virtual void op_call(VMContext* ctx, int pcount, void* instance) const; \
      )

FALCON_SYNCLASS_DECLARATOR_EX(m_expr_value, Value, Expression, \
      virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;\
      virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const; )

FALCON_SYNCLASS_DECLARATOR_EX(m_expr_autoclone, AutoClone, Expression, \
            virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;\
            virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const; )

FALCON_SYNCLASS_DECLARATOR_EX(m_expr_istring, IString, Expression, \
         virtual void store( VMContext*, DataWriter* dw, void* instance ) const;\
             )
   
   
   
   /** The normal behavior is that to unflatten all via nth.
   */
   
//======================================================================
// Statement classes
//
FALCON_SYNCLASS_DECLARATOR(m_stmt_break, Break, Statement)
FALCON_SYNCLASS_DECLARATOR(m_stmt_breakpoint, Breakpoint, Statement)
FALCON_SYNCLASS_DECLARATOR(m_stmt_continue, Continue, Statement)
FALCON_SYNCLASS_DECLARATOR(m_stmt_cut, Cut, Statement)
FALCON_SYNCLASS_DECLARATOR(m_stmt_doubt, Doubt, Statement)
FALCON_SYNCLASS_DECLARATOR(m_stmt_fastprint, FastPrint, Statement )
FALCON_SYNCLASS_DECLARATOR(m_stmt_fastprintnl, FastPrintNL, Statement )

FALCON_SYNCLASS_DECLARATOR_EX(m_stmt_forin, ForIn, Statement, \
         virtual void store( VMContext*, DataWriter* dw, void* instance ) const; \
       )

FALCON_SYNCLASS_DECLARATOR_EX(m_stmt_forto, ForTo, Statement, \
      virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;\
      virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const; )

FALCON_SYNCLASS_DECLARATOR(m_stmt_if, If, Statement)
FALCON_SYNCLASS_DECLARATOR(m_stmt_loop, Loop, Statement)
FALCON_SYNCLASS_DECLARATOR(m_stmt_raise, Raise, Statement)
FALCON_SYNCLASS_DECLARATOR_EX(m_stmt_global, Global, Statement, \
         virtual void store( VMContext*, DataWriter* dw, void* instance ) const; \
       )

FALCON_SYNCLASS_DECLARATOR_EX(m_stmt_return, Return, Statement,
         virtual void store( VMContext*, DataWriter* dw, void* instance ) const; \
         )

FALCON_SYNCLASS_DECLARATOR(m_stmt_rule, Rule, Expression)
FALCON_SYNCLASS_DECLARATOR(m_stmt_select, Select, Statement )
FALCON_SYNCLASS_DECLARATOR(m_stmt_switch, Switch, Statement)
FALCON_SYNCLASS_DECLARATOR(m_stmt_try, Try, Statement)
FALCON_SYNCLASS_DECLARATOR(m_stmt_while, While, Statement)

//======================================================================
// Syntree classes
//
FALCON_SYNCLASS_DECLARATOR(m_st_rulest, RuleSynTree, TreeStep)
