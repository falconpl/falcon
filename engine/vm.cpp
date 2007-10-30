/*
   FALCON - The Falcon Programming Language.
   FILE: vm.cpp
   $Id: vm.cpp,v 1.71 2007/08/18 12:07:37 jonnymind Exp $

   Implementation of virtual - non main loop
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-09-08
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/pcodes.h>
#include <falcon/runtime.h>
#include <falcon/vmcontext.h>
#include <falcon/sys.h>
#include <falcon/cobject.h>
#include <falcon/cclass.h>
#include <falcon/symlist.h>
#include <falcon/proptable.h>
#include <falcon/memory.h>
#include <falcon/stream.h>
#include <falcon/core_ext.h>
#include <falcon/stdstreams.h>
#include <falcon/traits.h>
#include <falcon/fassert.h>
#include <falcon/deferrorhandler.h>
#include <falcon/format.h>

#define VM_STACK_MEMORY_THRESHOLD 64

namespace Falcon {

VMachine::VMachine():
   m_services( &traits::t_string, &traits::t_voidp )
{
   internal_construct();
   init();
}

VMachine::VMachine( bool initItems ):
   m_services( &traits::t_string, &traits::t_voidp )
{
   internal_construct();
   if ( initItems )
      init();
}


void VMachine::internal_construct()
{
   m_userData = 0;
   m_bOwnErrorHandler = false;
   m_bhasStandardStreams = false;
   m_errhand =0;
   m_attributeCount = 0;
   m_error = 0;
   m_pc = 0;
   m_pc_next = 0;
   m_loopsGC = FALCON_VM_DFAULT_CHECK_LOOPS;
   m_loopsContext = FALCON_VM_DFAULT_CHECK_LOOPS;
   m_loopsCallback = 0;
   m_opLimit = 0;
   m_bSingleStep = false;
   m_sleepAsRequests = false;
   m_memPool = 0;
   m_stdIn = 0;
   m_stdOut = 0;
   m_stdErr = 0;
   m_tryFrame = i_noTryFrame;

   resetCounters();

   // this initialization must be performed by all vms.
   m_symbol = 0;
   m_moduleId = 0;
   m_allowYield = true;
   m_opCount = 0;

   // This vectror has also context ownership -- when we remove a context here, it's dead
   m_contexts.deletor( ContextList_deletor );

   // finally we create the context (and the stack)
   m_currentContext = new VMContext( this );
   // ... and then we take onwership of the items in the context.
   m_currentContext->restore( this );

   // saving also the first context for accounting reasons.
   m_contexts.pushBack( m_currentContext );

   // After context mangling, we have a stack
   m_stack->threshHold( VM_STACK_MEMORY_THRESHOLD );


   m_opHandlers = (tOpcodeHandler *) memAlloc( FLC_PCODE_COUNT * sizeof( tOpcodeHandler ) );

   // This code is actually here for debug reasons. Opcode management should
   // be performed via a swtich in the end, but until the beta version, this
   // method allows to have a stack trace telling immediately which opcode
   // were served in case a problem arises.
   m_opHandlers[ P_END ] = opcodeHandler_END ;
   m_opHandlers[ P_NOP ] = opcodeHandler_NOP ;
   m_opHandlers[ P_PSHN] = opcodeHandler_PSHN;
   m_opHandlers[ P_RET ] = opcodeHandler_RET ;
   m_opHandlers[ P_RETA] = opcodeHandler_RETA;

   // Range 2: one parameter ops
   m_opHandlers[ P_PTRY] = opcodeHandler_PTRY;
   m_opHandlers[ P_LNIL] = opcodeHandler_LNIL;
   m_opHandlers[ P_RETV] = opcodeHandler_RETV;
   m_opHandlers[ P_FORK] = opcodeHandler_FORK;
   m_opHandlers[ P_BOOL] = opcodeHandler_BOOL;
   m_opHandlers[ P_GENA] = opcodeHandler_GENA;
   m_opHandlers[ P_GEND] = opcodeHandler_GEND;
   m_opHandlers[ P_PUSH] = opcodeHandler_PUSH;
   m_opHandlers[ P_PSHR] = opcodeHandler_PSHR;
   m_opHandlers[ P_POP ] = opcodeHandler_POP ;
   m_opHandlers[ P_JMP ] = opcodeHandler_JMP ;
   m_opHandlers[ P_INC ] = opcodeHandler_INC ;
   m_opHandlers[ P_DEC ] = opcodeHandler_DEC ;
   m_opHandlers[ P_NEG ] = opcodeHandler_NEG ;
   m_opHandlers[ P_NOT ] = opcodeHandler_NOT ;
   m_opHandlers[ P_TRAL] = opcodeHandler_TRAL;
   m_opHandlers[ P_IPOP] = opcodeHandler_IPOP;
   m_opHandlers[ P_XPOP] = opcodeHandler_XPOP;
   m_opHandlers[ P_GEOR] = opcodeHandler_GEOR;
   m_opHandlers[ P_TRY ] = opcodeHandler_TRY ;
   m_opHandlers[ P_JTRY] = opcodeHandler_JTRY;
   m_opHandlers[ P_RIS ] = opcodeHandler_RIS ;
   m_opHandlers[ P_BNOT] = opcodeHandler_BNOT;
   m_opHandlers[ P_NOTS] = opcodeHandler_NOTS;
   m_opHandlers[ P_PEEK] = opcodeHandler_PEEK;

   // Range3: Double parameter ops
   m_opHandlers[ P_LD  ] = opcodeHandler_LD  ;
   m_opHandlers[ P_LDRF] = opcodeHandler_LDRF;
   m_opHandlers[ P_ADD ] = opcodeHandler_ADD ;
   m_opHandlers[ P_SUB ] = opcodeHandler_SUB ;
   m_opHandlers[ P_MUL ] = opcodeHandler_MUL ;
   m_opHandlers[ P_DIV ] = opcodeHandler_DIV ;
   m_opHandlers[ P_MOD ] = opcodeHandler_MOD ;
   m_opHandlers[ P_POW ] = opcodeHandler_POW ;
   m_opHandlers[ P_ADDS] = opcodeHandler_ADDS;
   m_opHandlers[ P_SUBS] = opcodeHandler_SUBS;
   m_opHandlers[ P_MULS] = opcodeHandler_MULS;
   m_opHandlers[ P_DIVS] = opcodeHandler_DIVS;
   m_opHandlers[ P_MODS] = opcodeHandler_MODS;
   m_opHandlers[ P_POWS] = opcodeHandler_POWS;
   m_opHandlers[ P_BAND] = opcodeHandler_BAND;
   m_opHandlers[ P_BOR ] = opcodeHandler_BOR ;
   m_opHandlers[ P_BXOR] = opcodeHandler_BXOR;
   m_opHandlers[ P_ANDS] = opcodeHandler_ANDS;
   m_opHandlers[ P_ORS ] = opcodeHandler_ORS ;
   m_opHandlers[ P_XORS] = opcodeHandler_XORS;
   m_opHandlers[ P_GENR] = opcodeHandler_GENR;
   m_opHandlers[ P_EQ  ] = opcodeHandler_EQ  ;
   m_opHandlers[ P_NEQ ] = opcodeHandler_NEQ ;
   m_opHandlers[ P_GT  ] = opcodeHandler_GT  ;
   m_opHandlers[ P_GE  ] = opcodeHandler_GE  ;
   m_opHandlers[ P_LT  ] = opcodeHandler_LT  ;
   m_opHandlers[ P_LE  ] = opcodeHandler_LE  ;
   m_opHandlers[ P_IFT ] = opcodeHandler_IFT ;
   m_opHandlers[ P_IFF ] = opcodeHandler_IFF ;
   m_opHandlers[ P_CALL] = opcodeHandler_CALL;
   m_opHandlers[ P_INST] = opcodeHandler_INST;
   m_opHandlers[ P_ONCE] = opcodeHandler_ONCE;
   m_opHandlers[ P_LDV ] = opcodeHandler_LDV ;
   m_opHandlers[ P_LDP ] = opcodeHandler_LDP ;
   m_opHandlers[ P_TRAN] = opcodeHandler_TRAN;
   m_opHandlers[ P_UNPK] = opcodeHandler_UNPK;
   m_opHandlers[ P_SWCH] = opcodeHandler_SWCH;
   m_opHandlers[ P_HAS ] = opcodeHandler_HAS ;
   m_opHandlers[ P_HASN] = opcodeHandler_HASN;
   m_opHandlers[ P_GIVE] = opcodeHandler_GIVE;
   m_opHandlers[ P_GIVN] = opcodeHandler_GIVN;
   m_opHandlers[ P_IN  ] = opcodeHandler_IN  ;
   m_opHandlers[ P_NOIN] = opcodeHandler_NOIN;
   m_opHandlers[ P_PROV] = opcodeHandler_PROV;
   m_opHandlers[ P_STPS] = opcodeHandler_STPS;
   m_opHandlers[ P_STVS] = opcodeHandler_STVS;
   m_opHandlers[ P_AND ] = opcodeHandler_AND;
   m_opHandlers[ P_OR  ] = opcodeHandler_OR;
   m_opHandlers[ P_PASS ] = opcodeHandler_PASS;
   m_opHandlers[ P_PSIN ] = opcodeHandler_PSIN;

   // Range 4: ternary opcodes
   m_opHandlers[ P_STP ] = opcodeHandler_STP ;
   m_opHandlers[ P_STV ] = opcodeHandler_STV ;
   m_opHandlers[ P_LDVT] = opcodeHandler_LDVT;
   m_opHandlers[ P_LDPT] = opcodeHandler_LDPT;
   m_opHandlers[ P_STPR] = opcodeHandler_STPR;
   m_opHandlers[ P_STVR] = opcodeHandler_STVR;
   m_opHandlers[ P_TRAV] = opcodeHandler_TRAV;

   m_opHandlers[ P_FORI] = opcodeHandler_FORI;
   m_opHandlers[ P_FORN] = opcodeHandler_FORN;

   m_opHandlers[ P_SHL ] = opcodeHandler_SHL;
   m_opHandlers[ P_SHR ] = opcodeHandler_SHR;
   m_opHandlers[ P_SHLS] = opcodeHandler_SHLS;
   m_opHandlers[ P_SHRS] = opcodeHandler_SHRS;
   m_opHandlers[ P_LDVR] = opcodeHandler_LDVR;
   m_opHandlers[ P_LDPR] = opcodeHandler_LDPR;
   m_opHandlers[ P_LSB ] = opcodeHandler_LSB;
   m_opHandlers[ P_UNPS ] = opcodeHandler_UNPS;
   m_opHandlers[ P_SELE ] = opcodeHandler_SELE;
   m_opHandlers[ P_INDI ] = opcodeHandler_INDI;
   m_opHandlers[ P_STEX ] = opcodeHandler_STEX;
   m_opHandlers[ P_TRAC ] = opcodeHandler_TRAC;
   m_opHandlers[ P_WRT ] = opcodeHandler_WRT;
}



void VMachine::init()
{
   //================================
   // Preparing memory handling
   if ( m_memPool == 0 )
      m_memPool = new MemPool();

   m_memPool->setOwner( this );

   //================================
   // Preparing minimal input/output
   if ( m_stdIn == 0 )
      m_stdIn = stdInputStream();

   if ( m_stdOut == 0 )
      m_stdOut = stdOutputStream();

   if ( m_stdErr == 0 )
      m_stdErr = stdErrorStream();

   if ( m_errhand == 0 )
   {
      m_bOwnErrorHandler = true;
      m_errhand = new DefaultErrorHandler( m_stdErr );
   }
}



VMachine::~VMachine()
{
   // destroy all the global vectors.
   for( uint32 i = 0; i < m_globals.size(); ++i )
   {
      delete m_globals.vpat( i );
   }

   delete  m_memPool ;
   memFree( m_opHandlers );

   // decref all the modules
   for( uint32 mi = 0; mi < m_modules.size(); mi++ )
   {
      m_modules.moduleAt( mi )->decref();
   }

   if ( m_bOwnErrorHandler )
      delete m_errhand;

   // delete the owned error
   if( m_error != 0 ) {
      m_error->decref();
   }

   delete m_stdErr;
   delete m_stdIn;
   delete m_stdOut;
}

void VMachine::errorHandler( ErrorHandler *em, bool own )
{
   if ( m_bOwnErrorHandler )
      delete m_errhand;

   m_errhand = em;
   m_bOwnErrorHandler = own;
}

bool VMachine::unlinkUpTo( uint32 count )
{
   if ( m_modules.size() <= count )
      return false;

   uint32 i = m_modules.size() - 1;
   while( i >= count )
   {
      Module *mod = m_modules.moduleAt( i );
      // unlink all the modulue shared items;
      MapIterator symiter = mod->symbolTable().map().begin();
      while( symiter.hasCurrent() ) {
         const Symbol *sym = *(const Symbol **) symiter.currentValue();
         if ( sym->exported() ) {
            m_globalSyms.erase( &sym->name() );
         }

         symiter.next();
      }

      delete m_globals.vpat( i );
      mod->decref();
      i --;
   }

   m_modules.resize( count );
   m_globals.resize( count );

   // unlink means that probably will be a re-run.
   resetCounters();

   return true;
}


bool VMachine::link( Runtime *rt )
{
   // link all the modules in the runtime from first to last.
   // FIFO order is important.
   for( uint32 iter = 0; iter < rt->moduleVector()->size(); ++iter )
   {
      if (! link( rt->moduleVector()->moduleAt( iter ) ) )
         return false;
   }

   return true;
}

bool VMachine::link( Module *mod )
{
   ItemVector *globs;
   int32 modId;

   // first of all link the exported services.
   MapIterator svmap_iter = mod->getServiceMap().begin();
   while( svmap_iter.hasCurrent() )
   {
      if ( ! publishService( *(Service ** ) svmap_iter.currentValue() ) )
         return false;
      svmap_iter.next();
   }

   // we need to record the classes in the module as they have to be evaluated last.
   SymbolList modClasses;
   SymbolList modObjects;

   // then we always need the symbol table.
   SymbolTable *symtab = &mod->symbolTable();

   // Ok, the module is now in.
   // We can now increment reference count ...
   mod->incref();
   // and add it to ourselves
   modId = m_modules.size();
   m_modules.push( mod );

   globs = new ItemVector;
   // resize() creates a series of NIL items.
   globs->resize( symtab->size()+1 );
   m_globals.push( globs );

   bool success = true;
   // now, the symbol table must be traversed.
   MapIterator iter = symtab->map().begin();
   while( iter.hasCurrent() )
   {
      Symbol *sym = *(Symbol **) iter.currentValue();
      if ( ! sym->isUndefined() )
      {
         // create an appropriate item here.
         switch( sym->type() )
         {
            case Symbol::tfunc:
            case Symbol::textfunc:
               globs->itemAt( sym->itemId() ).setFunction( sym, modId );
            break;

            case Symbol::tclass:
               modClasses.pushBack( sym );
            break;

            case Symbol::tattribute:
               if( m_attributeCount > FALCON_MAX_ATTRIBUTES )
               {
                  raiseError( new CodeError( ErrorParam( e_attrib_space, sym->declaredAt() ).
                     origin( e_orig_vm ).
                     symbol( sym->name() ).
                     module( sym->module()->name() ) )
                  );
                  return false;
               }
               else
               {
                  globs->itemAt( sym->itemId() ).setInteger( 1 << m_attributeCount++ );
               }
            break;

            case Symbol::tinst:
            {
               modObjects.pushBack( sym );
            }
            break;

            case Symbol::tvar:
            case Symbol::tconst:
            {
               Item *itm = globs->itemPtrAt( sym->itemId() );
               VarDef *vd = sym->getVarDef();
               switch( vd->type() ) {
                  case VarDef::t_int: itm->setInteger( vd->asInteger() ); break;
                  case VarDef::t_num: itm->setNumeric( vd->asNumeric() ); break;
                  case VarDef::t_string:
                  {
                     itm->setString( new GarbageString( this, *vd->asString() ) );
                  }
                  break;
               }
            }
            break;
         }

         // Is this symbol exported?
         if ( sym->exported() ) {
            // as long as the module is referenced, the symbols are alive, and as we
            // hold a reference to the module, we are sure that symbols are alive here.
            // also, in case an entry already exists, the previous item is just overwritten.

            if ( m_globalSyms.find( &sym->name() ) != 0 ) {

               raiseError(
                  new CodeError( ErrorParam( e_already_def, sym->declaredAt() ).origin( e_orig_vm ).
                        module( mod->name() ).
                        symbol( sym->name() ) )
                  );
               return false;
            }

            SymModule tmp( globs->itemPtrAt( sym->itemId() ), modId, sym );
            m_globalSyms.insert( &sym->name(), &tmp );

            // export also the instancer, if it is not already exported.
            if ( sym->isInstance() ) {
               sym = sym->getInstance();
               if ( ! sym->exported() ) {
                  SymModule tmp( globs->itemPtrAt( sym->itemId() ), modId, sym );
                  m_globalSyms.insert( &sym->name(), &tmp );
               }
            }
         }
      }
      else
      {
         // try to find the imported symbol.
         SymModule *sm = (SymModule *) m_globalSyms.find( &sym->name() );
         if( sm != 0 )
         {
            // link successful, we must set the current item as a reference of the original
            Symbol *sym2 = sm->symbol();
            referenceItem( globs->itemAt( sym->itemId() ), *sm->item() );
         }
         else {
            // raise undefined symbol.
            Error *error = new CodeError(
                  ErrorParam( e_undef_sym, sym->declaredAt() ).origin( e_orig_vm ).
                  module( mod->name() ).
                  extra( sym->name() )
                  );

            raiseError( error );
            // we're doomed to fail
            success = false;
         }
      }

      // next symbol
      iter.next();
   }

   // exit if link failed.
   if ( ! success )
      return false;

   // now that the symbols in the module have been linked, link the classes.
   ListElement *cls_iter = modClasses.begin();
   while( cls_iter != 0 )
   {
      Symbol *sym = (Symbol *) cls_iter->data();
      CoreClass *cc = linkClass( modId, sym );

      // we need to add it anyhow to the GC to provoke its destruction at VM end.
      // and hey, you could always destroy symbols if your mood is so from falcon ;-)
      // dereference as other classes may have referenced this item1
      globs->itemAt( cc->symbol()->itemId() ).dereference()->setClass( cc );
      cls_iter = cls_iter->next();
   }

   // then, prepare the instances of standalone objects

   ListElement *obj_iter = modObjects.begin();
   while( obj_iter != 0 )
   {
      Symbol *obj = (Symbol *) obj_iter->data();
      fassert( obj->isInstance() );
      Symbol *cls = obj->getInstance();
      Item *clsItem = globs->itemAt( cls->itemId() ).dereference();
      if ( ! clsItem->isClass() ) {
         raiseError(
            new CodeError( ErrorParam( e_no_cls_inst, obj->declaredAt() ).origin( e_orig_vm ).
               symbol( obj->name() ).
               module( obj->module()->name() ) )
         );
         return false;
      }
      else {
         CoreObject *co = clsItem->asClass()->createInstance();
         globs->itemAt( obj->itemId() ).dereference()->setObject( co );
      }
      obj_iter = obj_iter->next();
   }

   // and eventually call their constructor
   m_currentGlobals = globs;
   m_moduleId = modId;
   m_code = mod->code();
   obj_iter = modObjects.begin();

   // In case we have some objects to link
   if( obj_iter !=  0 )
   {
      // save S1 and S2, or we won't be able to link in scripts
      Item oldS1 = m_regS1;
      Item oldS2 = m_regS2;

      while( obj_iter != 0 )
      {
         Symbol *obj = (Symbol *) obj_iter->data();
         Symbol *cls = obj->getInstance();
         if ( cls->getClassDef()->constructor() != 0 )
         {
            Item *clsItem = globs->itemAt( cls->itemId() ).dereference();
            m_regS1 = globs->itemAt( obj->itemId() ) ;
            m_event = eventNone;
            callItem( *clsItem, 0, e_callInst );
            if ( m_event == eventRisen )
               return false;
         }
         obj_iter = obj_iter->next();
      }

      // clear S1
      m_regS1 = oldS1;
      // and s2 for safety
      m_regS2 = oldS2;
   }

   return true;
}

PropertyTable *VMachine::createClassTemplate( int modId, const Map &pt )
{
   MapIterator iter = pt.begin();
   PropertyTable *table = new PropertyTable( pt.size() );

   while( iter.hasCurrent() )
   {
      VarDefMod *vdmod = *(VarDefMod **) iter.currentValue();
      VarDef *vd = vdmod->vd;
      Item itm;

      // create the instance
      switch( vd->type() )
      {
         case VarDef::t_nil:
            itm.setNil();
         break;

         case VarDef::t_int:
            itm.setInteger( vd->asInteger() );
         break;

         case VarDef::t_num:
            itm.setNumeric( vd->asNumeric() );
         break;

         case VarDef::t_string:
         {
            itm.setString( new GarbageString( this, *vd->asString() ) );
         }
         break;

         case VarDef::t_reference:
         case VarDef::t_base:
         {
            const Symbol *sym = vd->asSymbol();
            Item *ptr = m_globals.vat(vdmod->modId).itemPtrAt( sym->itemId() );
            referenceItem( itm, *ptr );
         }
         break;

         case VarDef::t_symbol:
         {
            Symbol *sym = const_cast< Symbol *>( vd->asSymbol() );
            // may be a function or an extfunc
            fassert( sym->isExtFunc() || sym->isFunction() );
            if ( sym->isExtFunc() || sym->isFunction() )
            {
               itm.setFunction( sym, vdmod->modId );
            }
         }
         break;
      }

      String *key = *(String **) iter.currentKey();
      table->appendSafe( key, itm );
      iter.next();
   }

   return table;
}

CoreClass *VMachine::linkClass( uint32 modId, Symbol *clssym )
{
   Map props( &traits::t_stringptr, &traits::t_voidp ) ;
   uint64 attribs = 0;
   if( ! linkSubClass( modId, clssym, props, attribs ) )
      return 0;

   CoreClass *cc = new CoreClass( this, clssym, modId, createClassTemplate( modId, props ) );
   Symbol *ctor = clssym->getClassDef()->constructor();
   if ( ctor != 0 ) {
      cc->constructor().setFunction( ctor, modId );
   }
   cc->attributes( attribs );

   // destroy the temporary vardef we have created
   MapIterator iter = props.begin();
   while( iter.hasCurrent() )
   {
      VarDefMod *value = *(VarDefMod **) iter.currentValue();
      delete value;
      iter.next();
   }
   return cc;
}

bool VMachine::linkSubClass( uint32 modId, const Symbol *clssym,
      Map &props, uint64 &attribs )
{
   // first sub-instantiates all the inheritances.
   ClassDef *cd = clssym->getClassDef();
   ListElement *from_iter = cd->inheritance().begin();
   const Module *class_module = clssym->module();

   while( from_iter != 0 )
   {
      const InheritDef *def = (const InheritDef *) from_iter->data();
      const Symbol *parent = def->base();

      // iterates in the parent. Where is it?
      // 1) in the same module or 2) in the global modules.
      if( parent->isClass() )
      {
         // we create the item anew instead of relying on the already linked item.
         if ( ! linkSubClass( modId, parent, props, attribs ) )
            return false;
      }
      else if ( parent->isUndefined() )
      {
         const SymModule *sym_mod = findGlobalSymbol( parent->name() );
         // Can't be 0, because we have already linked all the external
         // symbols
         fassert( sym_mod != 0 );

         parent = sym_mod->symbol();
         if ( ! parent->isClass() )
         {
            raiseError(
               new CodeError( ErrorParam( e_inv_inherit, clssym->declaredAt() ).origin( e_orig_vm ).
                  symbol( clssym->name() ).
                  module( class_module->name() ) )
                  );
            return false;
         }
         if ( ! linkSubClass( sym_mod->moduleId(), parent, props, attribs ) )
            return false;
      }
      else
      {
         raiseError( new CodeError( ErrorParam( e_inv_inherit, clssym->declaredAt() ).origin( e_orig_vm ).
                  symbol( clssym->name() ).
                  module( class_module->name() ) )
         );
         return false;
      }
      from_iter = from_iter->next();
   }

   // then copies the vardefs declared in this class.
   MapIterator iter = cd->properties().begin();
   while( iter.hasCurrent() )
   {
      String *key = *(String **) iter.currentKey();
      VarDefMod *value = new VarDefMod;
      value->vd = *(VarDef **) iter.currentValue();
      value->modId = modId;
      // TODO: define vardefvalue traits
      VarDefMod **oldValue = (VarDefMod **) props.find( key );
      if ( oldValue != 0 )
         delete *oldValue;
      //==========================
      props.insert( key, value );

      iter.next();
   }

   // and set attributes;
   ListElement *hiter = cd->has().begin();
   while( hiter != 0 )
   {
      const Symbol *sym = (const Symbol *) hiter->data();
      Item *hitem = m_globals.vat( modId ).itemAt( sym->itemId() ).dereference();
      if ( hitem->isInteger() ) {
         attribs |= hitem->asInteger();
      }
      else {
         raiseError( new CodeError(
            ErrorParam( e_no_attrib, sym->declaredAt() ).origin( e_orig_vm ).
            symbol( sym->name() ).
            module( sym->module()->name() ) )
            );
      }

      hiter = hiter->next();
   }

   hiter = cd->hasnt().begin();
   while( hiter != 0 )
   {
      const Symbol *sym = (const Symbol *) hiter->data();
      Item *hitem = m_globals.vat( modId ).itemAt( sym->itemId() ).dereference();
      if ( hitem->isInteger() ) {
         attribs &= ~hitem->asInteger();
      }
      else {
         raiseError( new CodeError(
            ErrorParam( e_no_attrib, sym->declaredAt() ).origin( e_orig_vm ).
            symbol( sym->name() ).
            module( sym->module()->name() ) )
            );
      }
      hiter = hiter->next();
   }

   return true;
}


bool VMachine::prepare( const String &startSym, uint32 paramCount )
{

   // we must have at least one module.
   uint32 mod_id;
   if( m_modules.empty() ) {
      // I don't want an assertion, as it may be removed in optimized compilation
      // while the calling app must still be warned.
      m_symbol = 0;
      return false;
   }

   Symbol *execSym = m_modules.moduleAt( m_modules.size() - 1 )->findGlobalSymbol( startSym );
   if ( execSym == 0 )
   {
      SymModule *it_global = (SymModule *) m_globalSyms.find( &startSym );
      if( it_global == 0 ) {
         m_symbol = 0;
         raiseError( new CodeError(
            ErrorParam( e_undef_sym ).origin( e_orig_vm ).extra( startSym ).
            symbol( "prepare" ).
            module( "core.vm" ) )
            );
         return false;
      }
      else {
         execSym =  it_global->symbol();
         mod_id = it_global->moduleId();
      }
   }
   else
      mod_id = m_modules.size()-1; // module position

   /** \todo allow to call classes at startup. Something like "all-classes" a-la-java */
   if ( ! execSym->isFunction() ) {
      raiseError( new CodeError(
            ErrorParam( e_non_callable, execSym->declaredAt() ).origin( e_orig_vm ).
            symbol( execSym->name() ).
            module( execSym->module()->name() ) )
            );
      return false;
   }

   // ok, let's setup execution environment.
   const Module *mod = execSym->module();
   m_code = mod->code();
	FuncDef *tg_def = execSym->getFuncDef();
   m_pc = tg_def->offset();
   m_symbol = execSym;
   fassert( mod_id < m_globals.size() );
   m_currentGlobals = m_globals.vpat( mod_id );
   m_moduleId = mod_id;

   // reset the VM to ready it for execution
   reset();

	// manage an internal function
   // ensure against optional parameters.
   if( paramCount < tg_def->params() )
   {
      m_stack->resize( tg_def->params() - paramCount );
      paramCount = tg_def->params();
   }

	// space for locals
   if ( tg_def->locals() > 0 )
      m_stack->resize( tg_def->locals() );
   return true;
}


void VMachine::reset()
{
   // first, the trivial resets.

   // reset counters
   resetCounters();
   resetEvent();
   // reset stackbase
   m_stackBase = 0;

   // clear the accounting of sleeping contexts.
   m_sleepingContexts.clear();

   if ( m_contexts.size() > 1 )
   {
	   // clear the contexts
	   m_contexts.clear();

	   // as our frame, stack and tryframe were in one of the contexts,
	   // they have been destroyed.
	   m_currentContext = new VMContext( this );

	   // ... and then we take onwership of the items in the context.
	   m_currentContext->restore( this );

	   // saving also the first context for accounting reasons.
	   m_contexts.pushBack( m_currentContext );
   }
   else
   {
	   m_stack->resize(0);
	   m_tryFrame = i_noTryFrame;
   }

}

const SymModule *VMachine::findGlobalSymbol( const String &name ) const
{
   return (SymModule *) m_globalSyms.find( &name );
}

void VMachine::raiseError( int code, const String &expl, int32 line )
{
   Error *err = new CodeError(
         ErrorParam( code, line ).origin( e_orig_vm ).hard().extra( expl )
      );

   // should not be necessary, asthe first error in m_error will stop the VM,
   // but we do it in case of programming errors in the modules to avoid leaks on
   // multiple raises.
   if ( m_error != 0 )
      m_error->decref();
   m_error = err;

   // of course, if we're not executing, there nothing to raise
   if( currentSymbol() == 0 )
   {
      if ( m_errhand )
         m_errhand->handleError( err );
   }
   else {
      fillErrorContext( err );
      m_event = eventRisen;
   }
}


void VMachine::raiseError( Error *err )
{
   if ( m_error != 0 )
      m_error->decref();
   m_error = err;

   // of course, if we're not executing, there nothing to raise
   if( currentSymbol() == 0 )
   {
      if ( m_errhand )
         m_errhand->handleError( err );
   }
   else {
      m_event = eventRisen;
   }
}

void VMachine::raiseRTError( Error *err )
{
   if ( m_error != 0 )
      m_error->decref();
   m_error = err;

   // give an origin
   err->origin( e_orig_runtime );
   // of course, if we're not executing, there nothing to raise
   if( currentSymbol() == 0 )
   {
      if ( m_errhand )
         m_errhand->handleError( err );
   }
   else {
      fillErrorContext( err );
      m_event = eventRisen;
   }
}

void VMachine::raiseModError( Error *err )
{
   if ( m_error != 0 )
      m_error->decref();
   m_error = err;

   // of course, if we're not executing, there nothing to raise
   if( currentSymbol() == 0 )
   {
      if ( m_errhand )
         m_errhand->handleError( err );
   }
   else {
      if ( err->module().size() == 0 )
         err->module( currentSymbol()->module()->name() );

      if ( err->symbol().size() == 0 )
         err->symbol( currentSymbol()->name() );

      fillErrorTraceback( *err );
      m_event = eventRisen;
   }
}

void VMachine::fillErrorTraceback( Error &error )
{
   const Symbol *csym = currentSymbol();
   if ( csym != 0 )
   {
      error.addTrace( csym->module()->name(), csym->name(),
         csym->module()->getLineAt( programCounter() ),
         programCounter() );
   }

   uint32 base = m_stackBase;

   while( base != 0 )
   {
      StackFrame &frame = *(StackFrame *) m_stack->at( base - VM_FRAME_SPACE );
      Symbol *sym = frame.m_symbol;
      if ( sym != 0 )
      { // possible when VM has not been initiated from main
         uint32 line = sym->module()->getLineAt( frame.m_call_pc );
         error.addTrace( sym->module()->name(), sym->name(), line, frame.m_call_pc );
      }

      base = frame.m_stack_base;
   }
}


void VMachine::handleError( Error *err )
{
   if ( m_error )
   {
      m_error->decref();
   }

   m_error = err;
   m_error->incref();
   m_event = eventRisen;

   // we got either pass the error to the script or to our error handler
   // if we have a valid try frame, the script will handle somewhere.
   if ( m_tryFrame == i_noTryFrame )
   {
      // we got to create a traceback for this error.
      fillErrorTraceback( *err );

      // and if no script is in charge, we must also notify it. Else, we'll
      // wait for the main loop to exit
      if( m_symbol == 0 && m_errhand != 0 )
      {
         m_errhand->handleError( err );
      }
   }

}


void VMachine::fillErrorContext( Error *err, bool filltb )
{
   if( currentSymbol() != 0 )
   {
      err->module( currentModule()->name() );
      err->symbol( currentSymbol()->name() );
      err->line( currentModule()->getLineAt( programCounter() ) );
      err->pcounter( programCounter() );
   }

    if ( filltb )
      fillErrorTraceback( *err );

}

bool VMachine::callItem( const Item &callable, int32 paramCount, e_callMode callMode )
{
   Symbol *target;
   Item oldsender;
   CoreObject *self = 0;
   uint16 targetModId;

   switch( callable.type() )
   {
      case FLC_ITEM_FBOM:
      {
         m_bomParams = paramCount;
         bool bomRet = callable.callBom( this );
         //m_stack->resize( m_stack->size() - paramCount );
         return bomRet;
      }

      case FLC_ITEM_METHOD:
         self = callable.asMethodObject();
         target = callable.asMethodFunction();
         targetModId = callable.asModuleId();
      break;

      case FLC_ITEM_FUNC:
         target = callable.asFunction();
         targetModId = callable.asModuleId();
      break;

      case FLC_ITEM_CLASS:
      {
         CoreClass *cls = callable.asClass();
         if( callMode != e_callNormal && callMode != e_callFrame ) {
            self = m_regS1.asObject();
         }
         else {
            self = cls->createInstance();
         }

         // if the class has not a constructor, we just set the item in A
         // and return
         if ( cls->constructor().isNil() ) {
            m_regA.setObject( self );
            // pop the stack
            m_stack->resize( m_stack->size() - paramCount );
            return true;
         }

         target = cls->constructor().asFunction();
         targetModId = cls->constructor().asModuleId();
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         CoreArray *arr = callable.asArray();

         if ( arr->length() != 0 )
         {
            const Item &carr = callable.asArray()->at(0);

            if ( carr.isFbom() || carr.isFunction() || 
               carr.isMethod() || carr.isClass() )
            {
               uint32 sizeNow = m_stack->size();

               for ( uint32 i = 1; i < arr->length(); i ++ )
               {
                  pushParameter( arr->at(i) );
               }
               
               for ( uint32 j = sizeNow - paramCount; 
                  j < sizeNow; j ++ )
               {
                  pushParameter( m_stack->itemAt( j ) );
               }

               bool ret = callItem( carr, arr->length()-1 + paramCount );
               if ( paramCount != 0 )
                  m_stack->resize( m_stack->size() - paramCount );
               return ret;
            }
         }

         if ( paramCount != 0 )
               m_stack->resize( m_stack->size() - paramCount );
         return false;
      }
      break;

      default:
         // non callableitem
         if ( paramCount != 0 )
            m_stack->resize( m_stack->size() - paramCount );
         return false;
   }

   if ( target->isFunction() )
   {
      // manage internal functions
      FuncDef *tg_def = target->getFuncDef();

      // manage an internal function
      // ensure against optional parameters.
      if( paramCount < tg_def->params() )
      {
         m_stack->resize( m_stack->size() + tg_def->params() - paramCount );
         paramCount = tg_def->params();
      }
   }

   // space for frame
   m_stack->resize( m_stack->size() + VM_FRAME_SPACE );
   StackFrame *frame = (StackFrame *) m_stack->at( m_stack->size() - VM_FRAME_SPACE );
   frame->header.type( FLC_ITEM_INVALID );
   frame->m_symbol = m_symbol;
   frame->m_ret_pc = m_pc_next;
   frame->m_call_pc = m_pc;
   frame->m_modId = m_moduleId;
   frame->m_param_count = (byte)paramCount;
   frame->m_stack_base = m_stackBase;
   frame->m_try_base = m_tryFrame;
   frame->m_break = false;
   frame->m_suspend = false;

   // now we can change the stack base
   m_stackBase = m_stack->size();

   // Save the stack frame
   if ( callMode == e_callInst || callMode == e_callInstFrame )
   {
      frame->m_initFrame = true;
   }
   else
   {
      frame->m_initFrame = false;
      frame->m_sender = m_regS2;
      m_regS2 = m_regS1;
      if ( self == 0 )
         m_regS1.setNil();
      else
         m_regS1.setObject( self );
   }

   if ( target->isFunction() )
   {
      // manage internal functions
      FuncDef *tg_def = target->getFuncDef();
      // space for locals
      if ( tg_def->locals() > 0 )
         m_stack->resize( m_stackBase + tg_def->locals() );

      if( targetModId != m_moduleId )
      {
         m_code = target->module()->code();
         m_moduleId = targetModId;
         m_currentGlobals = m_globals.vpat( m_moduleId );
      }
      m_symbol = target;

      //jump
      m_pc_next = tg_def->offset();

      // If the function is not called internally by the VM, another run is issued.
      if( callMode == e_callNormal || callMode == e_callInst ) {
         // hitting the stack limit forces the RET code to raise a return event,
         // and this forces the machine to exit run().

         m_pc = m_pc_next;
         frame->m_break = true;
         if ( m_event == eventSuspend )
            frame->m_suspend = true;
         m_event = eventNone;
         run();
      }
   }
   else
   {
      // extfunc:
      // the PC may change dramatically if the extfunc uses the VM to call an item.
      // so we save it and restore it after the call.
      m_symbol = target; // so we can have adequate tracebacks.
      target->getExtFuncDef()->call( this );
      if ( callable.isClass() )
         m_regA.setObject( self );
      callReturn();
   }

   return true;
}


bool VMachine::callItemPass( const Item &callable  )
{
   Symbol *target;
   FuncDef *tg_def;
   CoreObject *self = 0;
   uint32 targetModId;

   switch( callable.type() )
   {
      case FLC_ITEM_FBOM:
      {
         m_bomParams = paramCount();
         bool bomRet = callable.callBom( this );
         callReturn();
         return bomRet;
      }

      case FLC_ITEM_METHOD:
         self = callable.asMethodObject();
         target = callable.asMethodFunction();
         targetModId = callable.asModuleId();
      break;

      case FLC_ITEM_FUNC:
         target = callable.asFunction();
         targetModId = callable.asModuleId();
      break;

      default:
         // non callableitem
         return false;
   }

   if ( target->isFunction() )
   {
      tg_def = target->getFuncDef();
      // manage an internal function
      // ensure against optional parameters.

      m_stack->resize( m_stackBase );
      StackFrame &frame = *(StackFrame*) m_stack->at( m_stackBase - VM_FRAME_SPACE );

      if ( frame.m_param_count < tg_def->params() )
      {
         uint32 oldCount = frame.m_param_count;
         uint32 oldBase = m_stackBase;
         m_stackBase += tg_def->params() - frame.m_param_count;
         frame.m_param_count = (byte)tg_def->params();
         m_stack->resize( m_stackBase + VM_FRAME_SPACE ); // fr no more valid

         // now copy the frame in the right position
         StackFrame &newPos = *(StackFrame*)m_stack->at( m_stackBase - VM_FRAME_SPACE );
         newPos = frame;
         for ( ; oldCount < newPos.m_param_count; oldCount++ )
         {
            m_stack->itemAt( m_stackBase - VM_FRAME_SPACE - newPos.m_param_count + oldCount ).setNil();
         }
      }

      // space for locals
      m_stack->resize( m_stackBase + VM_FRAME_SPACE + tg_def->locals() );

      if( m_moduleId != targetModId )
      {
         m_code = target->module()->code();
         m_moduleId = targetModId;
         m_currentGlobals = m_globals.vpat( m_moduleId );
      }
      m_symbol = target;

      //jump
      m_pc_next = tg_def->offset();

      return true;
   }

   else
   {
      m_stack->resize( m_stackBase );
      target->getExtFuncDef()->call( this );
      m_pc_next = ((StackFrame *)m_stack->at( m_stackBase - VM_FRAME_SPACE ))->m_ret_pc;

      callReturn();
   }
   return true;
}


bool VMachine::callItemPassIn( const Item &callable  )
{
   if( ! callable.isCallable() )
   {
      return false;
   }

   StackFrame *frame = (StackFrame *)m_stack->at( m_stackBase - VM_FRAME_SPACE );

   for( uint32 i = 0; i < frame->m_param_count; i++ )
      pushParameter( *param( i ) );

   return callItem( callable, frame->m_param_count, VMachine::e_callFrame );
}


void VMachine::yield( numeric secs )
{
   // be sure to allow yelding.
   m_allowYield = true;


   if ( m_sleepingContexts.empty() ) {
      if( secs >= 0.0 )
      {
         // should we ask the embedding application to wait for us??
         if ( m_sleepAsRequests )
         {
            m_event = eventSleep;
            m_yieldTime = secs;
         }
         else
         {
            // else just ask the system to sleep.
            Sys::_sleep( secs );
         }
      }
   }
   else if ( secs < 0.0 )
   {
      // terminate the context.
      opcodeHandler_END( this );
   }
   else
   {
      m_currentContext->save( this );
      // instead of just rotating, set a yielding sleep of 0
      putAtSleep( m_currentContext, secs );
      electContext();
   }
}

void VMachine::putAtSleep( VMContext *ctx, numeric secs )
{
   if ( secs < 0.0 ) {
      secs = 0.0;
   }

   numeric tgtTime = Sys::_seconds() + secs;
   ctx->schedule( tgtTime );
   ListElement *iter = m_sleepingContexts.begin();
   while( iter != 0 ) {
      VMContext *curctx = (VMContext *) iter->data();
      if ( tgtTime < curctx->schedule() ) {
         m_sleepingContexts.insertBefore( iter, ctx );
         return;
      }
      iter = iter->next();
   }

   // can't find it anywhere?
   m_sleepingContexts.pushBack( ctx );
}

void VMachine::rotateContext()
{
   m_currentContext->save( this );
   putAtSleep( m_currentContext, 0.0 );
   electContext();
}


void VMachine::electContext()
{
   // reset the event, so, in exit, we'll know if we have to wait.
   m_event = eventNone;

   // if there is some sleeping context...
   if ( ! m_sleepingContexts.empty() )
   {
      VMContext *elect = (VMContext *) m_sleepingContexts.front();
      numeric tgtTime = elect->schedule() - Sys::_seconds();
      if ( tgtTime > 0.0 )
      {

      // but eventually sleep.

         // should we ask the embedding application to wait for us??
         if ( m_sleepAsRequests )
         {
            m_event = eventSleep;
            m_yieldTime = tgtTime;
            return;
         }
         else
            Sys::_sleep( tgtTime );
      }

      // elect NOW the new context
      m_sleepingContexts.popFront();
      elect->restore( this );
      m_currentContext = elect;
      m_opCount = 0;
      // we must move to the next instruction after the context was swapped.
      m_pc = m_pc_next;
      // and restore the right globals
      m_currentGlobals = m_globals.vpat( m_moduleId );
   }
}


void VMachine::itemToString( String &target, const Item *itm, const String &format )
{
   if( itm->isObject() )
   {
      Item propString;
      if( itm->asObject()->getProperty( "toString", propString ) )
      {
         if ( propString.type() == FLC_ITEM_STRING )
            target = *propString.asString();
         else
         {
            Item old = m_regS1;
            m_regS1 = *itm;

            // eventually push parameters if format is required
            int params = 0;
            if( format.size() != 0 )
            {
               pushParameter( const_cast<String *>(&format) );
               params = 1;
            }

            bool success = callItem( propString, params, e_callInst );
            m_regS1 = old;
            if ( success ) {
               // if regA is already a string, it's a quite light operation.
               regA().toString( target );
            }
         }
      }
      else
         itm->toString( target );
   }
   else
      itm->toString( target );
}


void VMachine::callReturn()
{
   //... or we have nowhere to return.
   if( m_stackBase == 0 )
   {
      // do as if an end was in the code.
      // End will nil A, so we must save it
      Item oldA = m_regA;
      opcodeHandler_END( this );
      m_regA = oldA;
      return;
   }

   // fix the self and sender
   StackFrame &frame = *(StackFrame *) m_stack->at( m_stackBase - VM_FRAME_SPACE );
   if ( ! frame.m_initFrame ) {
      m_regS1 = m_regS2;
      m_regS2 = frame.m_sender;
   }

   // The caller may only be of type function; all the rest is managed before.
   m_stack->resize( m_stackBase - frame.m_param_count - VM_FRAME_SPACE );

   //get the stack frame.
   m_stackBase = frame.m_stack_base;
   if( frame.m_break )
   {
      m_event = frame.m_suspend ? eventSuspend : eventReturn;
   }

   // change symbol
   m_symbol = frame.m_symbol;
   m_pc_next = frame.m_ret_pc;

   // eventually change active module.

   register int32 modId = frame.m_modId;
   if ( modId != m_moduleId )
   {
      m_moduleId = modId;
      m_code = m_modules.moduleAt( modId )->code();
      m_currentGlobals = m_globals.vpat( modId );
   }

   // reset try frame
   m_tryFrame = frame.m_try_base;
}


bool VMachine::seekInteger( int64 value, byte *base, uint16 size, uint32 &landing ) const
{
   #undef SEEK_STEP
   #define SEEK_STEP (sizeof(int64) + sizeof(int32))

   fassert( size > 0 );  // should be granted before call
   int32 higher = size-1;
   byte *pos;

   int32 lower = 0;
   int32 point = higher / 2;

   while ( lower < higher - 1 )
   {
      pos = base + point * SEEK_STEP;

      if ( ((int64)endianInt64(*reinterpret_cast< int64 *>( pos ))) == value )
      {
         landing = endianInt32( *reinterpret_cast< int32 *>( pos + sizeof(int64) ) );
         return true;
      }

      if ( value > (int64) endianInt64(*reinterpret_cast< int64 *>( pos )) )
         lower = point;
      else
         higher = point;
      point = ( lower + higher ) / 2;
   }

   // see if it was in the last loop
   if ( ((int64)endianInt64( *reinterpret_cast< int64 *>( base + lower * SEEK_STEP ) ) ) == value )
   {
      landing =  endianInt32( *reinterpret_cast< uint32 *>( base + lower * SEEK_STEP + sizeof( int64 ) ) );
      return true;
   }

   if ( lower != higher && ((int64)endianInt64( *reinterpret_cast< int64 *>( base + higher * SEEK_STEP ) ) ) == value )
   {
      // YATTA, we found it at last
      landing =  endianInt32( *reinterpret_cast< uint32 *>( base + higher * SEEK_STEP + sizeof( int64 ) ) );
      return true;
   }

   return false;
}

bool VMachine::seekInRange( int64 numLong, byte *base, uint16 size, uint32 &landing ) const
{
   #undef SEEK_STEP
   #define SEEK_STEP (sizeof(int32) + sizeof(int32) + sizeof(int32))

   fassert( size > 0 );  // should be granted before call
   int32 higher = size-1;
   byte *pos;

   int32 value = (int32) numLong;
   int32 lower = 0;
   int32 point = higher / 2;

   while ( lower < higher - 1 )
   {
      pos = base + point * SEEK_STEP;

      if ( (int32)endianInt32(*reinterpret_cast< int32 *>( pos )) <= value &&
                (int32)endianInt32(*reinterpret_cast< int32 *>( pos + sizeof( int32 ) )) >= value)
      {
         landing = endianInt32( *reinterpret_cast< int32 *>( pos + sizeof( int32 ) + sizeof( int32 ) ) );
         return true;
      }

      if ( value > (int32) endianInt32(*reinterpret_cast< int32 *>( pos )) )
         lower = point;
      else
         higher = point;
      point = ( lower + higher ) / 2;
   }

   // see if it was in the last loop
   pos = base + lower * SEEK_STEP;
   if ( (int32)endianInt32( *reinterpret_cast< int32 *>( pos ) ) <= value &&
       (int32)endianInt32( *reinterpret_cast< int32 *>( pos + sizeof( int32 ) ) ) >= value )
   {
      landing =  endianInt32( *reinterpret_cast< uint32 *>( pos + sizeof( int32 ) + sizeof( int32 ) ) );
      return true;
   }

   if( lower != higher )
   {
      pos = base + higher * SEEK_STEP;
      if ( (int32)endianInt32( *reinterpret_cast< int32 *>( pos ) ) <= value &&
         (int32)endianInt32( *reinterpret_cast< int32 *>( pos + sizeof( int32 ) ) ) >= value )
      {
         // YATTA, we found it at last
         landing =  endianInt32( *reinterpret_cast< uint32 *>( pos + sizeof( int32 ) + sizeof( int32 ) ) );
         return true;
      }
   }

   return false;
}

bool VMachine::seekString( const String *value, byte *base, uint16 size, uint32 &landing ) const
{
   #undef SEEK_STEP
   #define SEEK_STEP (sizeof(int32) + sizeof(int32))

   fassert( size > 0 );  // should be granted before call
   int32 higher = size-1;
   byte *pos;

   int32 lower = 0;
   int32 point = higher / 2;
   const String *paragon;
   const Module *mod = m_modules.moduleAt( m_moduleId );

   while ( lower < higher - 1 )
   {
      pos = base + point * SEEK_STEP;
      paragon = mod->getString( endianInt32(*reinterpret_cast< int32 *>( pos )));
      fassert( paragon != 0 );
      if ( paragon == 0 )
         return false;
      if ( *paragon == *value )
      {
         landing = endianInt32( *reinterpret_cast< int32 *>( pos + sizeof(int32) ) );
         return true;
      }

      if ( *value > *paragon )
         lower = point;
      else
         higher = point;
      point = ( lower + higher ) / 2;
   }

   // see if it was in the last loop
   paragon = mod->getString( endianInt32(*reinterpret_cast< int32 *>( base + lower * SEEK_STEP )));
   if ( paragon != 0 && *paragon == *value )
   {
      landing =  endianInt32( *reinterpret_cast< uint32 *>( base + lower * SEEK_STEP + sizeof( int32 ) ) );
      return true;
   }

   if ( lower != higher )
   {
      paragon = mod->getString( endianInt32(*reinterpret_cast< int32 *>( base + higher * SEEK_STEP )));
      if ( paragon != 0 && *paragon == *value )
      {
         // YATTA, we found it at last
         landing =  endianInt32( *reinterpret_cast< uint32 *>( base + higher * SEEK_STEP + sizeof( int32 ) ) );
         return true;
      }
   }

   return false;
}

bool VMachine::seekItem( const Item *item, byte *base, uint16 size, uint32 &landing )
{
   #undef SEEK_STEP
   #define SEEK_STEP (sizeof(int32) + sizeof(int32))

   byte *target = base + size *SEEK_STEP;
   const Module *mod = m_modules.moduleAt( m_moduleId );

   while ( base < target )
   {
      Symbol *sym = mod->getSymbol( endianInt32( *reinterpret_cast< int32 *>( base ) ) );

      fassert( sym );
      if ( sym == 0 )
         return false;

      switch( sym->type() )
      {
         case Symbol::tlocal:
            if( compareItems( *stackItem( m_stackBase + VM_FRAME_SPACE +  sym->itemId() ).dereference(), *item ) == 0 )
               goto success;
         break;

         case Symbol::tparam:
            if( compareItems( *param( sym->itemId() ), *item ) )
               goto success;
         break;

         default:
            if( compareItems( *moduleItem( sym->itemId() ).dereference(), *item ) == 0 )
               goto success;
      }

      base += SEEK_STEP;
   }

   return false;

success:
   landing = endianInt32( *reinterpret_cast< int32 *>( base + sizeof(int32) ));
   return true;
}

bool VMachine::seekItemClass( const Item *itm, byte *base, uint16 size, uint32 &landing ) const
{
   #undef SEEK_STEP
   #define SEEK_STEP (sizeof(int32) + sizeof(int32))

   byte *target = base + size *SEEK_STEP;
   const Module *mod = m_modules.moduleAt( m_moduleId );

   while ( base < target )
   {
      Symbol *sym = mod->getSymbol( endianInt32( *reinterpret_cast< int32 *>( base ) ) );
      fassert( sym );
      if ( sym == 0 )
         return false;

      const Item *cfr;

      if ( sym->isLocal() )
      {
         cfr = stackItem( m_stackBase + VM_FRAME_SPACE +  sym->itemId() ).dereference();
      }
      else if ( sym->isParam() )
      {
         cfr = param( sym->itemId() );
      }
      else
      {
         cfr = moduleItem( sym->itemId() ).dereference();
      }

      switch( cfr->type() )
      {
         case FLC_ITEM_CLASS:
            if ( itm->isObject() )
            {
               const CoreObject *obj = itm->asObject();
               if ( obj->derivedFrom( cfr->asClass()->symbol()->name() ) )
                  goto success;
            }
            else if (itm->isClass() && itm->asClass() == cfr->asClass() )
            {
               goto success;
            }
         break;

         case FLC_ITEM_OBJECT:
            if ( itm->isObject() )
            {
               if( itm->asObject() == cfr->asObject() )
                  goto success;
            }
         break;

          case FLC_ITEM_INT:
            if ( cfr->asInteger() == itm->type() )
            {
               goto success;
            }
         break;

         case FLC_ITEM_STRING:
            if ( itm->isObject() && itm->asObject()->derivedFrom( *cfr->asString() ) )
               goto success;
         break;
      }

      base += SEEK_STEP;
   }

   return false;

success:
   landing = endianInt32( *reinterpret_cast< int32 *>( base + sizeof(int32) ));
   return true;
}

bool VMachine::publishService( Service *svr )
{
   Service **srv = (Service **) m_services.find( &svr->getServiceName() );
   if ( srv == 0 )
   {
      m_services.insert( &svr->getServiceName(), svr );
      return true;
   }
   else {
      raiseError( new CodeError(
            ErrorParam( e_service_adef ).origin( e_orig_vm ).
            extra( svr->getServiceName() ).
            symbol( "publishService" ).
            module( "core.vm" ) )
            );
      return false;
   }
}

Service *VMachine::getService( const String &name )
{
   Service **srv = (Service **) m_services.find( &name );
   if ( srv == 0 )
      return 0;
   return *srv;
}

int16 VMachine::getModuleId( const String modName )
{
   for ( uint32 i = 0; i < m_modules.size(); i ++ ) {
      if ( m_modules.moduleAt( i )->name() == modName )
         return i;
   }
   return -1;
}

void VMachine::stdIn( Stream *nstream )
{
   delete m_stdIn;
   m_stdIn = nstream;
}

void VMachine::stdOut( Stream *nstream )
{
   delete m_stdOut;
   m_stdOut = nstream;
}

void VMachine::stdErr( Stream *nstream )
{
   delete m_stdErr;
   m_stdErr = nstream;
}

void ContextList_deletor( void *data )
{
   VMContext *vmc = (VMContext *) data;
   delete vmc;
}

const String &VMachine::moduleString( uint32 stringId ) const
{
   static String empty;

   if ( currentModule() == 0 )
      return empty;

   const String *str = currentModule()->getString( stringId );
   if( str != 0 )
      return *str;

   return empty;
}

void VMachine::resetCounters()
{
   m_opCount = 0;
   m_opNextGC = m_loopsGC;
   m_opNextContext = m_loopsContext;
   m_opNextCallback = m_loopsCallback;

   m_opNextCheck = m_loopsGC < m_loopsContext ? m_loopsGC : m_loopsContext;
   if ( m_opNextCallback != 0 && m_opNextCallback < m_opNextCheck )
   {
      m_opNextCheck = m_opNextCallback;
   }
}

// basic implementation does nothing.
void VMachine::periodicCallback()
{}

void VMachine::pushTry( uint32 landingPC )
{
   // a small trick; we use a range to store current landing and previous try frame
   Item frame;


   frame.setRange( (int32) landingPC, (int32) m_tryFrame, false );
   m_tryFrame = m_stack->size();
   m_stack->push( &frame );
}

void VMachine::popTry( bool moveTo )
{
   // If the try frame is wrong or not in current stack frame...
   if ( m_stack->size() <= m_tryFrame || m_stackBase > m_tryFrame )
   {
      //TODO: raise proper error
      raiseError( new CodeError( ErrorParam( e_stackuf, m_symbol->declaredAt() ).
         origin( e_orig_vm ).
         symbol( m_symbol->name() ).
         module( m_modules.moduleAt(m_moduleId)->name() ) )
      );
      return;
   }

   // get the frame and resize the stack
   Item frame = m_stack->itemAt( m_tryFrame );
   m_stack->resize( m_tryFrame );

   // Change the try frame, and eventually move the PC to the proper position
   m_tryFrame = (uint32) frame.asRangeEnd();
   if( moveTo )
   {
      m_pc_next = (uint32) frame.asRangeStart();
      m_pc = m_pc_next;
   }
}

// TODO move elsewhere
inline bool vmIsWhiteSpace( uint32 chr )
{
   return chr == ' ' || chr == '\t' || chr == '\n' || chr == '\r';
}

inline bool vmIsTokenChr( uint32 chr )
{
   return chr >= 'A' || chr >= '0' && chr <= '9' || chr == '_';
}


Item *VMachine::findLocalSymbolItem( const String &symName ) const
{
   // parse self and sender
   if( symName == "self" )
   {
      return const_cast<Item *>(&self());
   }

   if( symName == "sender" )
   {
      return const_cast<Item *>(&sender());
   }

   // find the symbol
   const Symbol *sym = currentSymbol();
   if ( sym != 0 )
   {
      // get the relevant symbol table.
      const SymbolTable *symtab;
      if ( sym->isClass() )
      {
         symtab = &sym->getClassDef()->symtab();
      }
      else {
         symtab = &sym->getFuncDef()->symtab();
      }
      sym = symtab->findByName( symName );
   }


   // -- not a local symbol? -- try the global module table.
   if( sym == 0 )
   {
      sym = currentModule()->findGlobalSymbol( symName );
   }

   Item *itm = 0;
   if ( sym != 0 )
   {
      if ( sym->isLocal() )
      {
         itm = const_cast<VMachine *>(this)->local( sym->getItemId() )->dereference();
      }
      else if ( sym->isParam() )
      {
         itm = const_cast<VMachine *>(this)->param( sym->getItemId() )->dereference();
      }
      else {
         itm = const_cast<VMachine *>(this)->moduleItem( sym->getItemId() ).dereference();
      }
   }

   // if the item is zero, we didn't found it
   return itm;
}


Item *VMachine::findLocalVariable( const String &name ) const
{
   // item to be returned.
   Item *itm = 0;
   String sItemName;
   uint32 squareLevel = 0;
   uint32 len = name.length();

   typedef enum {
      initial,
      firstToken,
      interToken,
      dotAccessor,
      squareAccessor,
      postSquareAccessor,
      singleQuote,
      doubleQuote,
      strEscapeSingle,
      strEscapeDouble,
   } t_state;

   t_state state = initial;

   uint32 pos = 0;
   while( pos <= len )
   {
      // little trick: force a ' ' at len
      uint32 chr;
      if( pos == len )
         chr = ' ';
      else
         chr = name.getCharAt( pos );

      switch( state )
      {
         case initial:
            if( vmIsWhiteSpace( chr ) )
            {
               pos++;
               continue;
            }
            if( chr < 'A' )
               return 0;

            state = firstToken;
            sItemName.append( chr );
         break;

         //===================================================
         // Parse first token. It must be a valid local symbol
         case firstToken:
            if ( vmIsWhiteSpace( chr ) || chr == '.' || chr == '[' )
            {
               itm = findLocalSymbolItem( sItemName );

               // item not found?
               if( itm == 0 )
                  return 0;

               // set state accordingly to chr.
               goto resetState;
            }
            else if ( vmIsTokenChr( chr ) )
            {
               sItemName.append( chr );
            }
            else {
               // invalid format
               return 0;
            }
         break;

         //===================================================
         // Parse a dot accessor.
         //
         case dotAccessor:
            // wating for a complete token.

            if ( vmIsWhiteSpace( chr ) || chr == '.' || chr == '[' )
            {
               // ignore leading ws.
               if( sItemName.size() == 0 && vmIsWhiteSpace( chr ) )
                  break;

               // access the item. We know it's an object or we wouldn't be in this state.
               itm = itm->asObject()->getProperty( sItemName );
               if ( itm == 0 )
                  return 0;

               // set state accordingly to chr.
               goto resetState;
            }
            else if ( vmIsTokenChr( chr ) )
            {
               sItemName.append( chr );
            }
            else
               return 0;
         break;

         //===================================================
         // Parse the square accessor; from [ to matching ]

         case squareAccessor:
            // wating for complete square token.
            switch( chr )
            {
               case '[':
                  squareLevel++;
                  sItemName.append( chr );
               break;

               case ']':
                  if( --squareLevel == 0 )
                  {
                     itm = parseSquareAccessor( *itm, sItemName );
                     if( itm == 0 )
                        return 0;

                     goto resetState;
                  }
                  else
                     sItemName.append( chr );
               break;

               case '\'':
                  sItemName.append( chr );
                  state = singleQuote;
               break;

               case '"':
                  sItemName.append( chr );
                  state = doubleQuote;
               break;

               default:
                  sItemName.append( chr );
            }
         break;

         case postSquareAccessor:
            // wating for complete square token.
            if( chr == ']' )
            {
               if( --squareLevel == 0 )
               {
                  itm = parseSquareAccessor( *itm, sItemName );
                  if( itm == 0 )
                     return 0;

                  goto resetState;
               }
               else
                  sItemName.append( chr );
            }
            else if( ! vmIsWhiteSpace( chr ) )
            {
               return 0;
            }
         break;


         //===================================================
         // Parse the double quote inside suqare accessor
         case doubleQuote:
            switch( chr )
            {
               case '\\': state = strEscapeDouble; break;
               case '"': state = postSquareAccessor;  // do not break
               default:
                  sItemName.append( chr );
            }
         break;

         //===================================================
         // Parse the single quote inside suqare accessor
         case singleQuote:
            switch( chr )
            {
               case '\\': state = strEscapeSingle; break;
               case '\'': state = squareAccessor;  // do not break
               default:
                  sItemName.append( chr );
            }
         break;

         //===================================================
         // Parse the double quote inside suqare accessor
         case strEscapeDouble:
            sItemName.append( chr );
            state = doubleQuote;
         break;

         //===================================================
         // Parse the single quote inside suqare accessor
         case strEscapeSingle:
            sItemName.append( chr );
            state = singleQuote;
         break;

         //===================================================
         // Parse the space between tokens.
         case interToken:
            switch( chr ) {
               case '.':
                  if( ! itm->isObject() )
                     return 0;
                  state = dotAccessor;
               break;

               case '[':
                  if( ! itm->isDict() && ! itm->isArray() )
                     return 0;

                  state = squareAccessor;
                  squareLevel = 1;
               break;

               default:
                  if( ! vmIsWhiteSpace( chr ) )
                     return 0;
            }
         break;
      }

      // end the loop here
      pos++;
      continue;

      // state reset area.
resetState:

      sItemName.size(0); // clear sItemName.

      switch( chr ) {
         case '.':
            if( ! itm->isObject() )
               return 0;
            state = dotAccessor;
         break;

         case '[':
            if( ! itm->isDict() && ! itm->isArray() )
               return 0;

            state = squareAccessor;
            squareLevel = 1;
         break;

         default:
            state = interToken;
      }

      // end of loop, increment pos.
      pos++;
   }

   // if the state is not "interToken" we have an incomplete parse
   if( state != interToken )
      return 0;

   // Success
   return itm;
}


Item *VMachine::parseSquareAccessor( const Item &accessed, String &accessor ) const
{
   accessor.trim();

   // empty accessor? -- can't access!
   if( accessor.length() == 0)
      return 0;

   // what's the first character of the accessor?
   uint32 firstChar = accessor.getCharAt( 0 );

   // parse the accessor.
   Item acc;

   if( firstChar >= '0' && firstChar <= '9' )
   {
      // try to parse a number.
      int64 num;
      if( accessor.parseInt( num ) )
         acc = num;
      else
         return 0;
   }
   else if( firstChar == '\'' || firstChar == '"' )
   {
      // arrays cannot be accessed by strings.
      if( accessed.isArray() )
         return 0;

      accessor = accessor.subString( 1, accessor.length() - 1 );
      acc = &accessor;
   }
   else {
      // reparse the accessor as a token
      Item *parsed = findLocalVariable( accessor );
      if( parsed == 0 )
         return 0;
      acc = *parsed;
   }

   // what's the accessed item?
   if ( accessed.isDict() )
   {
      // find the accessor
      return accessed.asDict()->find( acc );
   }
   else if( accessed.isArray() )
   {
      // for arrays, only nubmbers and reaccessed items are
      if( !acc.isOrdinal() )
         return 0;

      uint32 pos = (uint32) acc.forceInteger();
      if(  pos >= accessed.asArray()->length() )
         return 0;

      return &accessed.asArray()->at( pos );
   }

   return 0;
}


VMachine::returnCode  VMachine::expandString( const String &src, String &target )
{
   uint32 pos0 = 0;
   uint32 pos1 = src.find( "$" );
   uint32 len = src.length();
   while( pos1 != String::npos )
   {
      target.append( src.subString( pos0, pos1 ) );
      pos1++;
      if( pos1 == len )
      {
         return return_error_string;
      }

      typedef enum {
         none,
         token,
         open,
         singleQuote,
         doubleQuote,
         escapeSingle,
         escapeDouble,
         complete,
         complete1,
         noAction,
         fail
      }
      t_state;

      t_state state = none;

      pos0 = pos1;
      uint32 chr = 0;
      while( pos1 < len && state != fail && state != complete && state != complete1 && state != noAction )
      {
         chr = src.getCharAt( pos1 );
         switch( state )
         {
            case none:
               if( chr == '$' )
               {
                  target.append( '$' );
                  state = noAction;
                  break;
               }
               else if ( chr == '(' )
               {
                  // scan for balanced ')'
                  pos0 = pos1+1;
                  state = open;
               }
               else if ( chr < '@' )
               {
                  state = fail;
               }
               else {
                  state = token;
               }
            break;

            case token:
               // allow also ':' and '|'
               if( (( chr < '0' && chr != '.' && chr != '%' ) || ( chr > ':' && chr <= '@' ))
                  && chr != '|' && chr != '[' && chr != ']' )
               {
                  state = complete;
                  pos1--;
                  // else we do this below.
               }
            break;

            case open:
               if( chr == ')' )
                  state = complete1;
               else if ( chr == '\'' )
                  state = singleQuote;
               else if ( chr == '\"' )
                  state = doubleQuote;
               // else just continue
            break;

            case singleQuote:
               if( chr == '\'' )
                  state = open;
               else if ( chr == '\\' )
                  state = escapeSingle;
               // else just continue
            break;

            case doubleQuote:
               if( chr == '\"' )
                  state = open;
               else if ( chr == '\\' )
                  state = escapeDouble;
               // else just continue
            break;

            case escapeSingle:
               state = singleQuote;
            break;

            case escapeDouble:
               state = escapeDouble;
            break;
         }

         ++pos1;
      }

      // parse the result in to the target.
      switch( state )
      {
         case token:
         case complete:
         case complete1:
         {
            uint32 pos2 = pos1;
            if( state == complete1 )
            {
               pos2--;
            }

            // todo: record this while scanning
            uint32 posColon = src.find( ":", pos0, pos2 );
            uint32 posPipe = src.find( "|", pos0, pos2 );
            Item *itm;

            if( posColon != String::npos ) {
               itm = findLocalVariable( src.subString( pos0, posColon ) );
            }
            else if( posPipe != String::npos )
            {
               // parse the format
               itm = findLocalVariable( src.subString( pos0, posPipe ) );
            }
            else {
               itm = findLocalVariable( src.subString( pos0, pos2 ) );
            }

            if ( itm == 0 )
            {
               return return_error_parse;
            }

            String temp;
            // do we have a format?
            if( posColon != String::npos )
            {
               Format fmt( src.subString( posColon+1, pos2 ) );
               if( ! fmt.isValid() )
               {
                  return return_error_parse_fmt;
               }

               if( ! fmt.format( this, *itm->dereference(), temp ) ) {
                  if( hadError() ) {
                     return return_error_internal;
                  }
                  return return_error_parse_fmt;
               }
            }
            // do we have a toString parameter?
            else if( posPipe != String::npos )
            {
               itemToString( temp, itm, src.subString( posPipe+1, pos2 ) );
            }
            else {
               // otherwise, add the toString version (todo format)
               // append to target.
               itemToString( temp, itm );

            }

            // having an error in an itemToString ?
            if( hadError() )
                  return return_error_internal;

            target.append( temp );
         }
         break;

         case noAction:
         break;

         default:
            return return_error_string;
      }

      pos0 = pos1;
      pos1 = src.find( "$", pos1 );
   }

   // add the last segment
   if( pos0 != pos1 )
   {
      target.append( src.subString( pos0, pos1 ) );
   }
   return return_ok;
}

int VMachine::compareItems( const Item &first, const Item &second )
{
   if ( first.isObject() )
   {
      CoreObject *fo = first.asObject();

      // provide a fast path. IF the items are the SAME object,
      // comparation is ==.
      if( second.isObject() && second.asObject() == fo )
         return 0;

      Item comparer;
      if( fo->getMethod( "compare", comparer ) )
      {
         Item oldA = m_regA;
         m_regA = (int64)0;

         pushParameter( second );
         callItem( comparer, 1 );
         // if the item is nil, fallback to normal item comparation.

         if ( hadError() )
         {
            m_regA = oldA;
            return 2;
         }

         if( ! m_regA.isNil() )
         {
            int val = (int) m_regA.forceInteger();
            m_regA = oldA;
            return val;
         }
         m_regA = oldA;
      }
      else if ( fo->getMethod( "equal", comparer ) )
      {
         Item oldA = m_regA;

         pushParameter( second );
         callItem( comparer, 1 );
         if ( hadError() )
         {
            m_regA = oldA;
            return 2;
         }

         if( m_regA.isTrue() )
         {
            m_regA = oldA;
            return 0;
         }
         m_regA = oldA;
         // else, fallback to standard item comparation.
      }
   }

   return first.compare( second );
}

void VMachine::referenceItem( Item &target, Item &source )
{
   if( source.isReference() ) {
      target.setReference( source.asReference() );
   }
   else {
      GarbageItem *itm = new GarbageItem( this, source );
      source.setReference( itm );
      target.setReference( itm );
   }
}

bool VMachine::functionalEval( const Item &itm )
{
   // Callitem includes a check for callability
   return callItem( itm, 0 ) ? m_regA.isTrue() : itm.isTrue();
}

}

/* end of vm.cpp */
