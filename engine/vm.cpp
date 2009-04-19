/*
   FALCON - The Falcon Programming Language.
   FILE: vm.cpp

   Implementation of virtual machine - non main loop
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-09-08

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/pcodes.h>
#include <falcon/runtime.h>
#include <falcon/vmcontext.h>
#include <falcon/sys.h>
#include <falcon/coreobject.h>
#include <falcon/cclass.h>
#include <falcon/corefunc.h>
#include <falcon/symlist.h>
#include <falcon/proptable.h>
#include <falcon/memory.h>
#include <falcon/stream.h>
#include <falcon/core_ext.h>
#include <falcon/stdstreams.h>
#include <falcon/traits.h>
#include <falcon/fassert.h>
#include <falcon/format.h>
#include <falcon/vm_sys.h>
#include <falcon/flexymodule.h>
#include <falcon/mt.h>
#include <falcon/vmmsg.h>
#include <falcon/livemodule.h>
#include <falcon/garbagelock.h>

#include <string.h>
namespace Falcon {

static ThreadSpecific s_currentVM;

VMachine *VMachine::getCurrent()
{
   return (VMachine *) s_currentVM.get();
}


VMachine::VMachine():
   m_services( &traits::t_string(), &traits::t_voidp() ),
   m_slots( &traits::t_string(), &traits::t_coreslotptr() ),
   m_nextVM(0),
   m_prevVM(0),
   m_idleNext( 0 ),
   m_idlePrev( 0 ),
   m_baton( this ),
   m_msg_head(0),
   m_msg_tail(0),
   m_refcount(1)
{
   internal_construct();
   init();
}

VMachine::VMachine( bool initItems ):
   m_services( &traits::t_string(), &traits::t_voidp() ),
   m_slots( &traits::t_string(), &traits::t_coreslotptr() ),
   m_nextVM(0),
   m_prevVM(0),
   m_idleNext( 0 ),
   m_idlePrev( 0 ),
   m_baton( this ),
   m_msg_head(0),
   m_msg_tail(0),
   m_refcount(1)
{
   internal_construct();
   if ( initItems )
      init();
}

void VMachine::incref()
{
   atomicInc( m_refcount );
}

void VMachine::decref()
{
   if( atomicDec( m_refcount ) == 0 )
   {
      delete this;
   }
}


void VMachine::setCurrent() const
{
   s_currentVM.set( (void*) this );
}

void VMachine::internal_construct()
{
   // use a ring for lock items.
   m_lockRoot = new GarbageLock(Item());
   m_lockRoot->next( m_lockRoot );
   m_lockRoot->prev( m_lockRoot );

   m_userData = 0;
   m_bhasStandardStreams = false;
   m_pc = 0;
   m_pc_next = 0;
   m_loopsGC = FALCON_VM_DFAULT_CHECK_LOOPS;
   m_loopsContext = FALCON_VM_DFAULT_CHECK_LOOPS;
   m_loopsCallback = 0;
   m_opLimit = 0;
   m_generation = 0;
   m_bSingleStep = false;
   m_sleepAsRequests = false;
   m_stdIn = 0;
   m_stdOut = 0;
   m_stdErr = 0;
   m_tryFrame = i_noTryFrame;
   m_launchAtLink = true;
   m_bGcEnabled = true;
   m_bWaitForCollect = false;
   m_bPirorityGC = false;


   resetCounters();

   // this initialization must be performed by all vms.
   m_symbol = 0;
   m_currentModule = 0;
   m_currentGlobals = 0;
   m_mainModule = 0;
   m_allowYield = true;
   m_atomicMode = false;
   m_opCount = 0;

   // This vectror has also context ownership -- when we remove a context here, it's dead
   m_contexts.deletor( ContextList_deletor );

   // finally we create the context (and the stack)
   m_currentContext = new VMContext( this );
   // ... and then we take onwership of the items in the context.
   m_currentContext->restore( this );

   // saving also the first context for accounting reasons.
   m_contexts.pushBack( m_currentContext );

   m_opHandlers = (tOpcodeHandler *) memAlloc( FLC_PCODE_COUNT * sizeof( tOpcodeHandler ) );

   m_metaClasses = (CoreClass**) memAlloc( FLC_ITEM_COUNT * sizeof(CoreClass*) );
   memset( m_metaClasses, 0, FLC_ITEM_COUNT * sizeof(CoreClass*) );

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
   m_opHandlers[ P_IN  ] = opcodeHandler_IN  ;
   m_opHandlers[ P_NOIN] = opcodeHandler_NOIN;
   m_opHandlers[ P_PROV] = opcodeHandler_PROV;
   m_opHandlers[ P_STPS] = opcodeHandler_STPS;
   m_opHandlers[ P_STVS] = opcodeHandler_STVS;
   m_opHandlers[ P_AND ] = opcodeHandler_AND;
   m_opHandlers[ P_OR  ] = opcodeHandler_OR;

   // Range 4: ternary opcodes
   m_opHandlers[ P_STP ] = opcodeHandler_STP ;
   m_opHandlers[ P_STV ] = opcodeHandler_STV ;
   m_opHandlers[ P_LDVT] = opcodeHandler_LDVT;
   m_opHandlers[ P_LDPT] = opcodeHandler_LDPT;
   m_opHandlers[ P_STPR] = opcodeHandler_STPR;
   m_opHandlers[ P_STVR] = opcodeHandler_STVR;
   m_opHandlers[ P_TRAV] = opcodeHandler_TRAV;

   m_opHandlers[ P_INCP] = opcodeHandler_INCP;
   m_opHandlers[ P_DECP] = opcodeHandler_DECP;

   m_opHandlers[ P_SHL ] = opcodeHandler_SHL;
   m_opHandlers[ P_SHR ] = opcodeHandler_SHR;
   m_opHandlers[ P_SHLS] = opcodeHandler_SHLS;
   m_opHandlers[ P_SHRS] = opcodeHandler_SHRS;
//   m_opHandlers[ P_LDVR] = opcodeHandler_LDVR;
//   m_opHandlers[ P_LDPR] = opcodeHandler_LDPR;
   m_opHandlers[ P_LSB ] = opcodeHandler_LSB;
   m_opHandlers[ P_UNPS ] = opcodeHandler_UNPS;
   m_opHandlers[ P_SELE ] = opcodeHandler_SELE;
   m_opHandlers[ P_INDI ] = opcodeHandler_INDI;
   m_opHandlers[ P_STEX ] = opcodeHandler_STEX;
   m_opHandlers[ P_TRAC ] = opcodeHandler_TRAC;
   m_opHandlers[ P_WRT ] = opcodeHandler_WRT;
   m_opHandlers[ P_STO ] = opcodeHandler_STO;
   m_opHandlers[ P_FORB ] = opcodeHandler_FORB;
   m_opHandlers[ P_EVAL ] = opcodeHandler_EVAL;
   m_opHandlers[ P_OOB ] = opcodeHandler_OOB;

   // Finally, register to the GC system
   memPool->registerVM( this );
}



void VMachine::init()
{
   //================================
   // Preparing minimal input/output
   if ( m_stdIn == 0 )
      m_stdIn = stdInputStream();

   if ( m_stdOut == 0 )
      m_stdOut = stdOutputStream();

   if ( m_stdErr == 0 )
      m_stdErr = stdErrorStream();
}


void VMachine::finalize()
{
   // we should have at least 2 refcounts here: one is from the caller and one in the GC.
   fassert( m_refcount >= 2 );

   // disengage from mempool
   if ( memPool != 0 )
   {
      memPool->unregisterVM( this );
   }

   decref();
}


VMachine::~VMachine()
{
   // Free generic tables (quite safe)
   memFree( m_opHandlers );
   memFree( m_metaClasses );

   // and finally, the streams.
   delete m_stdErr;
   delete m_stdIn;
   delete m_stdOut;

   // clear now the global maps
   // this also decrefs the modules and destroys the globals.
   // Notice that this would be done automatically also at destructor exit.
   m_liveModules.clear();

   // delete the garbage ring.
   GarbageLock *ge = m_lockRoot->next();
   while( ge != m_lockRoot )
   {
      GarbageLock *gnext = ge->next();
      delete ge;
      ge = gnext;
   }
   delete ge;

}


LiveModule* VMachine::link( Runtime *rt )
{
   // link all the modules in the runtime from first to last.
   // FIFO order is important.
   uint32 listSize = rt->moduleVector()->size();
   LiveModule* lmod = 0;
   for( uint32 iter = 0; iter < listSize; ++iter )
   {
      ModuleDep *md = rt->moduleVector()->moduleDepAt( iter );
      if ( (lmod = link( md->module(),
                       rt->hasMainModule() && (iter + 1 == listSize),
                       md->isPrivate() ) ) == 0
      )
      {
         return 0;
      }
   }

   // returns the topmost livemodule
   return lmod;
}


LiveModule *VMachine::link( Module *mod, bool isMainModule, bool bPrivate )
{
   // See if we have a module with the same name
   LiveModule *oldMod = findModule( mod->name() );
   if ( oldMod != 0 )
   {
      // if the publish policy is changed, allow this
      if( oldMod->isPrivate() && ! bPrivate )
      {
         // try to export all
         if ( ! exportAllSymbols( oldMod ) )
         {
            return 0;
         }
         // success; change official policy and return the livemod
         oldMod->setPrivate( true );
      }

      return oldMod;
   }

   // first of all link the exported services.
   MapIterator svmap_iter = mod->getServiceMap().begin();
   while( svmap_iter.hasCurrent() )
   {
      if ( ! publishService( *(Service ** ) svmap_iter.currentValue() ) )
         return false;
      svmap_iter.next();
   }

   // Ok, the module is now in.
   // We can now increment reference count and add it to ourselves
   LiveModule *livemod = new LiveModule( mod, bPrivate );
   // set this as the main module if required.
   if ( isMainModule )
      m_mainModule = livemod;

   if ( liveLink( livemod, lm_complete ) )
      return livemod;

   // no need to free on failure: livemod are garbaged
   livemod->mark( generation() );
   return 0;
}

LiveModule *VMachine::prelink( Module *mod, bool bIsMain, bool bPrivate )
{
   // See if we have a module with the same name
   LiveModule *oldMod = findModule( mod->name() );
   if ( oldMod == 0 )
   {
      oldMod = new LiveModule( mod, bPrivate );
      m_liveModules.insert( &oldMod->name(), oldMod );
      oldMod->mark( generation() );
   }

   if ( bIsMain )
      m_mainModule = oldMod;

   return oldMod;
}

bool VMachine::postlink()
{
   MapIterator iter = m_liveModules.begin();
   while( iter.hasCurrent() )
   {
      LiveModule *livemod = *(LiveModule **) iter.currentValue();

      if ( livemod->initialized() != LiveModule::init_complete )
      {
         if ( ! liveLink( livemod, lm_postlink ) )
            return false;
      }

      iter.next();
   }

   return true;
}


bool VMachine::liveLink( LiveModule *livemod, t_linkMode mode )
{
   if ( mode == lm_postlink )
   {
      // mark the module as being inspected
      livemod->initialized( LiveModule::init_trav );

      MapIterator deps = livemod->module()->dependencies().begin();
      while( deps.hasCurrent() )
      {
         const ModuleDepData *depdata = *(const ModuleDepData **) deps.currentValue();
         const String *moduleName = depdata->moduleName();

         // have we got that module?
         LiveModule *needed = findModule( *moduleName );
         if( needed == 0 )
         {
            // not present
            return false;
         }
         else if( needed->initialized() == LiveModule::init_trav )
         {
            // ricular ref
            return false;
         }
         else if( needed->initialized() == LiveModule::init_none )
         {
            // must link THAT before us
            if ( ! liveLink( needed, mode ) )
               return false;
         }
         // else the module is already linked in.

         // have we got the module?
         deps.next();
      }
   }


   // we need to record the classes in the module as they have to be evaluated last.
   SymbolList modClasses;
   SymbolList modObjects;

   // then we always need the symbol table.
   const SymbolTable *symtab = &livemod->module()->symbolTable();

   // A shortcut
   ItemVector *globs = &livemod->globals();

   // resize() creates a series of NIL items.
   globs->resize( symtab->size()+1 );

   // we won't be preemptible during link
   bool atomic = m_atomicMode;
   m_atomicMode = true;

   bool success = true;
   // now, the symbol table must be traversed.
   MapIterator iter = symtab->map().begin();
   while( iter.hasCurrent() )
   {
      Symbol *sym = *(Symbol **) iter.currentValue();

      if ( linkSymbol( sym, livemod ) )
      {
         // save classes and objects for later linking.
         if( sym->type() == Symbol::tclass )
            modClasses.pushBack( sym );
         else if ( sym->type() == Symbol::tinst )
            modObjects.pushBack( sym );
      }
      else {
         // but continue to expose other errors as well.
         success = false;
      }

      // next symbol
      iter.next();
   }

   // now that the symbols in the module have been linked, link the classes.
   ListElement *cls_iter = modClasses.begin();
   while( cls_iter != 0 )
   {
      Symbol *sym = (Symbol *) cls_iter->data();
      fassert( sym->isClass() );

      // on error, report failure but proceed.
      if ( ! linkClassSymbol( sym, livemod ) )
         success = false;

      cls_iter = cls_iter->next();
   }

   // then, prepare the instances of standalone objects
   ListElement *obj_iter = modObjects.begin();
   while( obj_iter != 0 )
   {
      Symbol *obj = (Symbol *) obj_iter->data();
      fassert( obj->isInstance() );

      // on error, report failure but proceed.
      if ( ! linkInstanceSymbol( obj, livemod ) )
         success = false;

      obj_iter = obj_iter->next();
   }

   // eventually, call the constructors declared by the instances
   obj_iter = modObjects.begin();

   // In case we have some objects to link - and while we have no errors,
   // -- we can't afford calling constructors if everything is not ok.
   while( success && obj_iter != 0 )
   {
      Symbol *obj = (Symbol *) obj_iter->data();
      initializeInstance( obj, livemod );
      obj_iter = obj_iter->next();
   }

   // Initializations of module objects is complete; return to non-atomic mode
   m_atomicMode = atomic;

   // return zero and dispose of the module if not succesful.
   if ( ! success )
   {
      // LiveModule is garbageable, cannot be destroyed.
      return false;
   }

   // We can now add the module to our list of available modules.
   m_liveModules.insert( &livemod->name(), livemod );
   livemod->initialized( LiveModule::init_complete );
   livemod->mark( generation() );

   // execute the main code, if we have one
   // -- but only if this is NOT the main module
   if ( m_launchAtLink && m_mainModule != livemod )
   {
      Item *mainItem = livemod->findModuleItem( "__main__" );
      if( mainItem != 0 )
      {
         callItem( *mainItem, 0 );
      }
   }

   return true;
}


// Link a single symbol
bool VMachine::linkSymbol( const Symbol *sym, LiveModule *livemod )
{
   // A shortcut
   ItemVector *globs = &livemod->globals();
   const Module *mod = livemod->module();

   if ( sym->isUndefined() )
   {
      // is the symbol name-spaced?
      uint32 dotPos;

      String localSymName;
      ModuleDepData *depData;
      LiveModule *lmod = 0;

      if ( ( dotPos = sym->name().rfind( "." ) ) != String::npos )
      {
         String nameSpace = sym->name().subString( 0, dotPos );
         // get the module name for the given module
         depData = mod->dependencies().findModule( nameSpace );
         // if we linked it, it must exist
         fassert( depData != 0 );

         // ... then find the module in the item
         lmod = findModule( Module::absoluteName(
               *depData->moduleName(), mod->name() ));

         // we must convert the name if it contains self or if it starts with "."
         if ( lmod != 0 )
            localSymName = sym->name().subString( dotPos + 1 );
      }
      else if ( sym->isImportAlias() )
      {
         depData = mod->dependencies().findModule( *sym->getImportAlias()->origModule() );
         // if we linked it, it must exist
         fassert( depData != 0 );

         // ... then find the module in the item
         lmod = findModule( Module::absoluteName(
               *depData->moduleName(), mod->name() ));

         if( lmod != 0 )
            localSymName = *sym->getImportAlias()->name();
      }

      // If we found it it...
      if ( lmod != 0 )
      {
         Symbol *localSym = lmod->module()->findGlobalSymbol( localSymName );

         if ( localSym != 0 )
         {
            referenceItem( globs->itemAt( sym->itemId() ),
               lmod->globals().itemAt( localSym->itemId() ) );
            return true;
         }

         // last chance: if the module is flexy, we may ask it do dynload it.
         if( lmod->module()->isFlexy() )
         {
            // Destroy also constness; flexy modules love to be abused.
            FlexyModule *fmod = (FlexyModule *)( lmod->module() );
            Symbol *newsym = fmod->onSymbolRequest( localSymName );

            // Found -- great, link it and if all it's fine, link again this symbol.
            if ( newsym != 0 )
            {
               // be sure to allocate enough space in the module global table.
               if ( newsym->itemId() >= lmod->globals().size() )
               {
                  lmod->globals().resize( newsym->itemId() );
               }

               // now we have space to link it.
               if ( linkCompleteSymbol( newsym, lmod ) )
               {
                  referenceItem( globs->itemAt( sym->itemId() ), *lmod->globals().itemPtrAt( newsym->itemId() ) );
                  return true;
               }
               else {
                  // we found the symbol, but it was flacky. We must have raised an error,
                  // and so we should return now.
                  // Notice that there is no need to free the symbol.
                  return false;
               }
            }
         }
         // ... otherwise, the symbol is undefined.
      }
      else {
         // try to find the imported symbol.
         SymModule *sm = (SymModule *) m_globalSyms.find( &sym->name() );

         if( sm != 0 )
         {
            // link successful, we must set the current item as a reference of the original
            referenceItem( globs->itemAt( sym->itemId() ), *sm->item() );
            return true;
         }
      }

      // try to dynamically load the symbol from flexy modules.
      SymModule symmod;
      if ( linkSymbolDynamic( sym->name(), symmod ) )
      {
         referenceItem( globs->itemAt( sym->itemId() ), *symmod.item() );
         return true;
      }

      // We failed every try; raise undefined symbol.
      Error *error = new CodeError(
            ErrorParam( e_undef_sym, sym->declaredAt() ).origin( e_orig_vm ).
            module( mod->name() ).
            extra( sym->name() )
            );

      raiseError( error );
      return false;
   }

   // Ok, the symbol is defined here. Link (record) it.

   // create an appropriate item here.
   // NOTE: Classes and instances are handled separately.
   switch( sym->type() )
   {
      case Symbol::tfunc:
      case Symbol::textfunc:
         globs->itemAt( sym->itemId() ).setFunction( new CoreFunc( sym, livemod ) );
      break;

      case Symbol::tvar:
      case Symbol::tconst:
      {
         Item *itm = globs->itemPtrAt( sym->itemId() );
         VarDef *vd = sym->getVarDef();
         switch( vd->type() ) {
            case VarDef::t_bool: itm->setBoolean( vd->asBool() ); break;
            case VarDef::t_int: itm->setInteger( vd->asInteger() ); break;
            case VarDef::t_num: itm->setNumeric( vd->asNumeric() ); break;
            case VarDef::t_string:
            {
               itm->setString( new CoreString( *vd->asString() ) );
            }
            break;

            default:
               break;
         }
      }
      break;

      // nil when we don't know what it is.
      default:
         globs->itemAt( sym->itemId() ).setNil();
   }

   // see if the symbol needs exportation and eventually do that.
   if ( ! exportSymbol( sym, livemod ) )
      return false;

   return true;
}


bool VMachine::exportAllSymbols( LiveModule *livemod )
{
   bool success = true;

   // now, the symbol table must be traversed.
   const SymbolTable *symtab = &livemod->module()->symbolTable();
   MapIterator iter = symtab->map().begin();
   while( iter.hasCurrent() )
   {
      Symbol *sym = *(Symbol **) iter.currentValue();

      if ( ! exportSymbol( sym, livemod ) )
      {
         // but continue to expose other errors as well.
         success = false;
      }

      // next symbol
      iter.next();
   }

   return success;
}


bool VMachine::exportSymbol( const Symbol *sym, LiveModule *livemod )
{
   // A shortcut
   ItemVector *globs = &livemod->globals();
   const Module *mod = livemod->module();

      // Is this symbol exported?
   if ( ! livemod->isPrivate() && sym->exported() && sym->name().getCharAt(0) != '_' )
   {
      // as long as the module is referenced, the symbols are alive, and as we
      // hold a reference to the module, we are sure that symbols are alive here.
      // also, in case an entry already exists, the previous item is just overwritten.

      if ( m_globalSyms.find( &sym->name() ) != 0 )
      {
         raiseError(
            new CodeError( ErrorParam( e_already_def, sym->declaredAt() ).origin( e_orig_vm ).
                  module( mod->name() ).
                  symbol( sym->name() ) )
            );
         return false;
      }

      SymModule tmp( globs->itemPtrAt( sym->itemId() ), livemod, sym );
      m_globalSyms.insert( &sym->name(), &tmp );

      // export also the instance, if it is not already exported.
      if ( sym->isInstance() )
      {
         sym = sym->getInstance();
         if ( ! sym->exported() ) {
            SymModule tmp( globs->itemPtrAt( sym->itemId() ), livemod, sym );
            m_globalSyms.insert( &sym->name(), &tmp );
         }
      }
   }

   // Is this symbol a well known item?
   if ( sym->isWKS() )
   {
      if ( m_wellKnownSyms.find( &sym->name() ) != 0 )
      {
         raiseError(
            new CodeError( ErrorParam( e_already_def, sym->declaredAt() ).origin( e_orig_vm ).
                  module( mod->name() ).
                  symbol( sym->name() ).
                  extra( "Well Known Item" ) )
            );
         return false;
      }

      SymModule tmp( livemod->wkitems().size(), livemod, sym );
      m_wellKnownSyms.insert( &sym->name(), &tmp );

      // and don't forget to add a copy of the item
      livemod->wkitems().push( globs->itemPtrAt( sym->itemId() ) );
   }

   return true;
}


bool VMachine::linkSymbolDynamic( const String &name, SymModule &symdata )
{
   // For now, the thing is very unoptimized; we'll traverse all the live modules,
   // and see which of them is flexy.
   MapIterator iter = m_liveModules.begin();
   while( iter.hasCurrent() )
   {
      LiveModule *lmod = *(LiveModule **)(iter.currentValue());

      if( lmod->module()->isFlexy() )
      {
         // Destroy also constness; flexy modules love to be abused.
         FlexyModule *fmod = (FlexyModule *)( lmod->module() );
         Symbol *newsym = fmod->onSymbolRequest( name );

         // Found -- great, link it and if all it's fine, link again this symbol.
         if ( newsym != 0 )
         {
            // be sure to allocate enough space in the module global table.
            if ( newsym->itemId() >= lmod->globals().size() )
            {
               lmod->globals().resize( newsym->itemId() );
            }

            // now we have space to link it.
            if ( linkCompleteSymbol( newsym, lmod ) )
            {
               symdata = SymModule( lmod->globals().itemPtrAt( newsym->itemId() ), lmod, newsym );
               return true;
            }
            else {
               // we found the symbol, but it was flacky. We must have raised an error,
               // and so we should return now.
               // Notice that there is no need to free the symbol.
               return false;
            }
         }
         // otherwise, go on
      }

      iter.next();
   }

   // sorry, not found.
   return false;
}

bool VMachine::linkClassSymbol( const Symbol *sym, LiveModule *livemod )
{
   // shortcut
   ItemVector *globs = &livemod->globals();

   CoreClass *cc = linkClass( livemod, sym );
   if ( cc == 0 )
      return false;

   // we need to add it anyhow to the GC to provoke its destruction at VM end.
   // and hey, you could always destroy symbols if your mood is so from falcon ;-)
   // dereference as other classes may have referenced this item1
   globs->itemAt( cc->symbol()->itemId() ).dereference()->setClass( cc );

   // if this class was a WKI, we must also set the relevant exported symbol
   if ( sym->isWKS() )
   {
      SymModule *tmp = (SymModule *) m_wellKnownSyms.find( &sym->name() );
      fassert( tmp != 0 ); // we just added it
      tmp->liveModule()->wkitems().itemAt( tmp->wkiid() ) = cc;
   }

   if ( sym->getClassDef()->isMetaclassFor() >= 0 )
   {
      m_metaClasses[ sym->getClassDef()->isMetaclassFor() ] = cc;
   }

   return true;
}


bool VMachine::linkInstanceSymbol( const Symbol *obj, LiveModule *livemod )
{
   // shortcut
   ItemVector *globs = &livemod->globals();
   Symbol *cls = obj->getInstance();
   Item *clsItem = globs->itemAt( cls->itemId() ).dereference();

   if ( clsItem == 0 || ! clsItem->isClass() ) {
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

      // if this class was a WKI, we must also set the relevant exported symbol
      if ( obj->isWKS() )
      {
         SymModule *tmp = (SymModule *) m_wellKnownSyms.find( &obj->name() );
         fassert( tmp != 0 ); // we just added it
         tmp->liveModule()->wkitems().itemAt( tmp->wkiid() ) = co;
      }
   }

   return true;
}


void VMachine::initializeInstance( const Symbol *obj, LiveModule *livemod )
{
   ItemVector *globs = &livemod->globals();

   Symbol *cls = obj->getInstance();
   if ( cls->getClassDef()->constructor() != 0 )
   {
      SafeItem ctor = *globs->itemAt( cls->getClassDef()->constructor()->itemId() ).dereference();
      ctor.methodize( *globs->itemAt( obj->itemId() ).dereference() );

      // If we can't call, we have a wrong init.
      try {
         callItemAtomic( ctor, 0 );
      }
      catch( Error *err )
      {
         err->extraDescription( "_init" );
         err->origin( e_orig_vm );
         throw;
      }
   }
}


bool VMachine::linkCompleteSymbol( const Symbol *sym, LiveModule *livemod )
{
   // try a pre-link
   bool bSuccess = linkSymbol( sym, livemod );

   // Proceed anyhow, even on failure, for classes and instance symbols
   if( sym->type() == Symbol::tclass )
   {
      if ( ! linkClassSymbol( sym, livemod ) )
         bSuccess = false;
   }
   else if ( sym->type() == Symbol::tinst )
   {
      fassert( sym->getInstance() != 0 );

      // we can't try to call the initialization method
      // if the creation of the symbol fails.
      if ( linkClassSymbol( sym->getInstance(), livemod ) &&
           linkInstanceSymbol( sym, livemod )
      )
      {
         initializeInstance( sym, livemod );
      }
      else
         bSuccess = false;
   }

   return bSuccess;
}


bool VMachine::linkCompleteSymbol( Symbol *sym, const String &moduleName )
{
   LiveModule *lm = findModule( moduleName );
   if ( lm != 0 )
      return linkCompleteSymbol( sym, lm );

   return false;
}


PropertyTable *VMachine::createClassTemplate( LiveModule *lmod, const Map &pt )
{
   MapIterator iter = pt.begin();
   PropertyTable *table = new PropertyTable( pt.size() );

   while( iter.hasCurrent() )
   {
      VarDefMod *vdmod = *(VarDefMod **) iter.currentValue();
      VarDef *vd = vdmod->vd;

      String *key = *(String **) iter.currentKey();
      PropEntry &e = table->appendSafe( key );

      e.m_bReadOnly = vd->isReadOnly();

      // configure the element
      if ( vd->isReflective() )
      {
         e.m_eReflectMode = vd->asReflecMode();
         e.m_reflection.offset = vd->asReflecOffset();
      }
      else if ( vd->isReflectFunc() )
      {
         e.m_eReflectMode = e_reflectFunc;
         e.m_reflection.rfunc.to = vd->asReflectFuncTo();
         e.m_reflection.rfunc.from = vd->asReflectFuncFrom();
         e.reflect_data = vd->asReflectFuncData();

         // just to be paranoid
         if( e.m_reflection.rfunc.to == 0 )
            e.m_bReadOnly = true;
      }

      // create the instance
      switch( vd->type() )
      {
         case VarDef::t_nil:
            e.m_value.setNil();
         break;

         case VarDef::t_bool:
            e.m_value.setBoolean( vd->asBool() );
         break;

         case VarDef::t_int:
            e.m_value.setInteger( vd->asInteger() );
         break;

         case VarDef::t_num:
            e.m_value.setNumeric( vd->asNumeric() );
         break;

         case VarDef::t_string:
         {
            e.m_value.setString( new CoreString( *vd->asString() ) );
         }
         break;

         case VarDef::t_base:
            e.m_bReadOnly = true;
         case VarDef::t_reference:
         {
            const Symbol *sym = vd->asSymbol();
            Item *ptr = vdmod->lmod->globals().itemPtrAt( sym->itemId() );
            referenceItem( e.m_value, *ptr );

         }
         break;

         case VarDef::t_symbol:
         {
            Symbol *sym = const_cast< Symbol *>( vd->asSymbol() );
            // may be a function or an extfunc
            fassert( sym->isExtFunc() || sym->isFunction() );
            if ( sym->isExtFunc() || sym->isFunction() )
            {
               e.m_value.setFunction( new CoreFunc( sym, vdmod->lmod ) );
            }
         }
         break;

         default:
            break; // compiler warning no-op
      }

      iter.next();
   }

   table->checkProperties();

   return table;
}


CoreClass *VMachine::linkClass( LiveModule *lmod, const Symbol *clssym )
{
   Map props( &traits::t_stringptr(), &traits::t_voidp() ) ;

   ObjectFactory factory = 0;
   if( ! linkSubClass( lmod, clssym, props, &factory ) )
      return 0;

   CoreClass *cc = new CoreClass( clssym, lmod, createClassTemplate( lmod, props ) );
   Symbol *ctor = clssym->getClassDef()->constructor();
   if ( ctor != 0 ) {
      cc->constructor().setFunction( new CoreFunc( ctor, lmod ) );
   }

   // destroy the temporary vardef we have created
   MapIterator iter = props.begin();
   while( iter.hasCurrent() )
   {
      VarDefMod *value = *(VarDefMod **) iter.currentValue();
      delete value;
      iter.next();
   }

   // ok, now determine the default object factory, if not provided.
   if( factory != 0 )
   {
      cc->factory( factory );
   }
   else
   {
      if ( ! cc->properties().isReflective() )
      {
         // a standard falcon object
         cc->factory( FalconObjectFactory );
      }
      else
      {
         if ( cc->properties().isStatic() )
         {
            // A fully reflective class.
            cc->factory( ReflectFalconFactory );
         }
         else
         {
            // a partially reflective class.
            cc->factory( CRFalconFactory );
         }
      }
   }

   return cc;
}


bool VMachine::linkSubClass( LiveModule *lmod, const Symbol *clssym,
      Map &props, ObjectFactory *factory )
{
   // first sub-instantiates all the inheritances.
   ClassDef *cd = clssym->getClassDef();
   ListElement *from_iter = cd->inheritance().begin();
   const Module *class_module = clssym->module();

   // If the class is final, we're doomed, as this is called on subclasses
   if( cd->isFinal() )
   {
      raiseError(
         new CodeError( ErrorParam( e_final_inherit, clssym->declaredAt() ).origin( e_orig_vm ).
            symbol( clssym->name() ).
            module( class_module->name() ) )
            );
      return false;
   }

   if( *factory != 0 && cd->factory() != 0 )
   {
      // raise an error for duplicated object manager.
      raiseError(
         new CodeError( ErrorParam( e_inv_inherit2, clssym->declaredAt() ).origin( e_orig_vm ).
            symbol( clssym->name() ).
            module( class_module->name() ) )
            );
      return false;
   }

   ObjectFactory subFactory = 0;

   while( from_iter != 0 )
   {
      const InheritDef *def = (const InheritDef *) from_iter->data();
      const Symbol *parent = def->base();

      // iterates in the parent. Where is it?
      // 1) in the same module or 2) in the global modules.
      if( parent->isClass() )
      {
         // we create the item anew instead of relying on the already linked item.
         if ( ! linkSubClass( lmod, parent, props, &subFactory ) )
            return false;
      }
      else if ( parent->isUndefined() )
      {
         // we have already linked the symbol for sure.
         Item *icls = lmod->globals().itemAt( parent->itemId() ).dereference();

         if ( ! icls->isClass() )
         {
            raiseError(
               new CodeError( ErrorParam( e_inv_inherit, clssym->declaredAt() ).origin( e_orig_vm ).
                  symbol( clssym->name() ).
                  module( class_module->name() ) )
                  );
            return false;
         }

         parent = icls->asClass()->symbol();
         LiveModule *parmod = findModule( parent->module()->name() );
         if ( ! linkSubClass( parmod, parent, props, &subFactory ) )
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

   // assign our manager
   if ( cd->factory() != 0 )
      *factory = cd->factory();
   else
      *factory = subFactory;

   // then copies the vardefs declared in this class.
   MapIterator iter = cd->properties().begin();
   while( iter.hasCurrent() )
   {
      String *key = *(String **) iter.currentKey();
      VarDefMod *value = new VarDefMod;
      value->vd = *(VarDef **) iter.currentValue();
      value->lmod = lmod;
      // TODO: define vardefvalue traits
      VarDefMod **oldValue = (VarDefMod **) props.find( key );
      if ( oldValue != 0 )
         delete *oldValue;
      //==========================
      props.insert( key, value );

      iter.next();
   }

   return true;
}


bool VMachine::prepare( const String &startSym, uint32 paramCount )
{

   // we must have at least one module.
   LiveModule *curMod;

   if( m_mainModule == 0 ) {
      // I don't want an assertion, as it may be removed in optimized compilation
      // while the calling app must still be warned.
      m_symbol = 0;
      return false;
   }

   const Symbol *execSym = m_mainModule->module()->findGlobalSymbol( startSym );
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
         curMod = it_global->liveModule();
      }
   }
   else
      curMod = m_mainModule; // module position

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
   //const Module *mod = execSym->module();
	FuncDef *tg_def = execSym->getFuncDef();
	m_code = tg_def->code();
   m_pc = 0;
   m_symbol = execSym;
   m_currentGlobals = &curMod->globals();
   m_currentModule = curMod;

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

   // of course, if we're not executing, there nothing to raise
   if( currentSymbol() != 0 )
   {
      fillErrorContext( err );
      m_event = eventRisen;
   }

   throw err;
}


void VMachine::raiseError( Error *err )
{
   // of course, if we're not executing, there nothing to raise
   if( currentSymbol() != 0 )
   {
      fillErrorContext( err );
      m_event = eventRisen;
   }

   throw err;
}

void VMachine::raiseRTError( Error *err )
{
   // give an origin
   err->origin( e_orig_runtime );
   // of course, if we're not executing, there nothing to raise
   if( currentSymbol() != 0 )
   {
      fillErrorContext( err );
      m_event = eventRisen;
   }

   throw err;
}

void VMachine::raiseModError( Error *err )
{
   // of course, if we're not executing, there nothing to raise
   if( currentSymbol() != 0 )
   {
      if ( err->module().size() == 0 )
         err->module( currentSymbol()->module()->name() );

      if ( err->symbol().size() == 0 )
         err->symbol( currentSymbol()->name() );

      fillErrorTraceback( *err );
      m_event = eventRisen;
   }

   throw err;
}

void VMachine::fillErrorTraceback( Error &error )
{
   fassert( ! error.hasTraceback() );

   const Symbol *csym = currentSymbol();
   if ( csym != 0 )
   {
      uint32 curLine;
      if( csym->isFunction() )
      {
         curLine = csym->module()->getLineAt( csym->getFuncDef()->basePC() + programCounter() );
      }
      else {
         // should have been filled by raise
         curLine = error.line();
      }

      error.addTrace( csym->module()->name(), csym->name(),
         curLine,
         programCounter() );
   }

   uint32 base = m_stackBase;

   while( base != 0 )
   {
      StackFrame &frame = *(StackFrame *) m_stack->at( base - VM_FRAME_SPACE );
      const Symbol *sym = frame.m_symbol;
      if ( sym != 0 )
      { // possible when VM has not been initiated from main
         uint32 line;
         if( sym->isFunction() )
            line = sym->module()->getLineAt( sym->getFuncDef()->basePC() + frame.m_call_pc );
         else
            line = 0;

         error.addTrace( sym->module()->name(), sym->name(), line, frame.m_call_pc );
      }

      base = frame.m_stack_base;
   }
}




bool VMachine::getCaller( const Symbol *&sym, const Module *&module)
{
   if ( m_stackBase < VM_FRAME_SPACE )
      return false;

   StackFrame &frame = *(StackFrame *) m_stack->at( m_stackBase - VM_FRAME_SPACE );
   sym = frame.m_symbol;
   module = frame.m_module->module();
   return sym != 0 && module != 0;
}

bool VMachine::getCallerItem( Item &caller, uint32 level )
{
   uint32 sbase = m_stackBase;
   while( sbase >= VM_FRAME_SPACE && level > 0 )
   {
      StackFrame &frame = *(StackFrame *) m_stack->at( sbase - VM_FRAME_SPACE );
      sbase = frame.m_stack_base;
      level--;
   }

   if ( sbase < VM_FRAME_SPACE )
      return false;

   StackFrame &frame = *(StackFrame *) m_stack->at( sbase - VM_FRAME_SPACE );
   const Symbol* sym = frame.m_symbol;
   const LiveModule* module = frame.m_module;
   caller = module->globals().itemAt( sym->itemId() );
   if ( ! caller.isFunction() )
      return false;

   // was it a method ?
   if ( ! frame.m_self.isNil() )
   {
      caller.methodize( frame.m_self );
   }

   return true;
}

void VMachine::fillErrorContext( Error *err, bool filltb )
{
   if( currentSymbol() != 0 )
   {
      if ( err->module().size() == 0 )
         err->module( currentModule()->name() );

      if ( err->module().size() == 0 )
         err->symbol( currentSymbol()->name() );

      if( m_symbol->isFunction() )
         err->line( currentModule()->getLineAt( m_symbol->getFuncDef()->basePC() + programCounter() ) );

      err->pcounter( programCounter() );
   }

    if ( filltb )
      fillErrorTraceback( *err );

}


void VMachine::createFrame( uint32 paramCount )
{
   // space for frame
   m_stack->resize( m_stack->size() + VM_FRAME_SPACE );
   StackFrame *frame = (StackFrame *) m_stack->at( m_stack->size() - VM_FRAME_SPACE );
   frame->header.type( FLC_ITEM_INVALID );
   frame->m_symbol = m_symbol;
   frame->m_ret_pc = m_pc_next;
   frame->m_call_pc = m_pc;
   frame->m_module = m_currentModule;
   frame->m_param_count = (byte)paramCount;
   frame->m_stack_base = m_stackBase;
   frame->m_try_base = m_tryFrame;
   frame->m_break = false;
   frame->m_binding = m_regBind;
   frame->m_self = m_regS1;

   // iterative processing support
   frame->m_endFrameFunc = 0;

   // now we can change the stack base
   m_stackBase = m_stack->size();
}


void VMachine::callFrameNow( ext_func_frame_t callbackFunc )
{
   ((StackFrame *)m_stack->at( m_stackBase - VM_FRAME_SPACE ) )->m_endFrameFunc = callbackFunc;
   switch( m_pc )
   {
      case i_pc_call_external_ctor:
         m_pc_next = i_pc_call_external_ctor_return;
         break;
      case i_pc_call_external:
         m_pc_next = i_pc_call_external_return;
         break;
      default:
         m_pc_next = m_pc;
   }
}


void VMachine::callItemAtomic(const Item &callable, int32 paramCount )
{
   bool oldAtomic = m_atomicMode;
   m_atomicMode = true;
   callFrame( callable, paramCount );
   execFrame();
   m_atomicMode = oldAtomic;
}



void VMachine::returnHandler( ext_func_frame_t callbackFunc )
{
   if ( m_stackBase >= VM_FRAME_SPACE )
   {
      StackFrame *frame = (StackFrame *) m_stack->at( m_stackBase - VM_FRAME_SPACE );
      frame->m_endFrameFunc = callbackFunc;
   }
}


ext_func_frame_t VMachine::returnHandler()
{
   if ( m_stackBase > VM_FRAME_SPACE )
   {
      StackFrame *frame = (StackFrame *) m_stack->at( m_stackBase - VM_FRAME_SPACE );
      return frame->m_endFrameFunc;
   }
   return 0;
}

void VMachine::yield( numeric secs )
{
   if ( m_atomicMode )
   {
      throw new InterruptedError( ErrorParam( e_wait_in_atomic ).origin( e_orig_vm ).
            symbol( "yield" ).
            module( "core.vm" ).
            line( __LINE__ ).
            hard() );
   }

   // be sure to allow yelding.
   m_allowYield = true;

   if ( m_sleepingContexts.empty() )
   {
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
            idle();
            if ( ! m_systemData.sleep( secs ) )
            {
               m_systemData.resetInterrupt();
               unidle();

               fassert( m_symbol->isFunction() || m_symbol->isExtFunc() );

               raiseError( new InterruptedError(
                  ErrorParam( e_interrupted ).origin( e_orig_vm ).
                     symbol( m_symbol->name() ).
                     module( m_currentModule->name() ).
                     line( m_currentModule->module()->getLineAt( m_symbol->getFuncDef()->basePC() + m_pc ) )
                  ) );
            }

            unidle();
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
   if ( secs < 0.0 )
   {
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


void VMachine::reschedule( VMContext *ctx, numeric secs )
{
   numeric tgtTime = Sys::_seconds() + secs;
   ctx->schedule( tgtTime );
   ListElement *iter = m_sleepingContexts.begin();

   bool bFound = false;
   while( iter != 0 )
   {
      VMContext *curctx = (VMContext *) iter->data();

      // if the rescheduled context is in sleeping context,
      // signal we've found it.
      if ( curctx == ctx )
      {
         ListElement *old = iter;
         iter = iter->next();
         m_sleepingContexts.erase( old );
         continue;
      }

      if ( tgtTime < curctx->schedule() )
      {
         m_sleepingContexts.insertBefore( iter, ctx );
         bFound = true;
      }
      iter = iter->next();
   }

   // can't find it anywhere?
   if ( ! bFound )
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

      // elect NOW the new context
      m_sleepingContexts.popFront();
      elect->wakeup();
      elect->restore( this );

      m_currentContext = elect;
      m_opCount = 0;
      // we must move to the next instruction after the context was swapped.
      m_pc = m_pc_next;

      // but eventually sleep.
      if ( tgtTime > 0.0 )
      {
         // should we ask the embedding application to wait for us??
         if ( m_sleepAsRequests )
         {
            m_event = eventSleep;
            m_yieldTime = tgtTime;
            return;
         }
         else {
            // raise an interrupted error on need.
            idle();
            if ( ! m_systemData.sleep( tgtTime ) )
            {
               m_systemData.resetInterrupt();
               unidle();

               //fassert( m_symbol->isFunction() );
               raiseError( new InterruptedError(
                  ErrorParam( e_interrupted ).origin( e_orig_vm ).
                     symbol( m_symbol->name() ).
                     module( m_currentModule->name() ).
                     line( currentModule()->getLineAt( m_symbol->getFuncDef()->basePC() + m_pc ) )
                  ) );
            }
            unidle();
         }
      }
   }
}


void VMachine::terminateCurrentContext()
{
   // scan the contexts and remove the current one.
   if ( m_sleepingContexts.empty() )
   {
      // there is wating non-sleeping context that will never be awaken?
      if( m_contexts.size() != 1 )
      {
         raiseRTError( new CodeError( ErrorParam( e_deadlock ).extra("END").origin( e_orig_vm ) ) );
         return;
      }

      m_event = eventQuit;
   }
   else
   {
      ListElement *iter = m_contexts.begin();
      while( iter != 0 ) {
         if( iter->data() == m_currentContext ) {
            m_contexts.erase( iter );
               // removing the context also deletes it.

               // Not necessary, but we do for debug reasons (i.e. if we access it before election, we crash)
               m_currentContext = 0;

               break;
         }
         iter = iter->next();
      }

      electContext();
   }
}


void VMachine::itemToString( String &target, const Item *itm, const String &format )
{
   if( itm->isObject() )
   {
      Item propString;
      if( itm->asObjectSafe()->getMethod( "toString", propString ) )
      {
         if ( propString.type() == FLC_ITEM_STRING )
            target = *propString.asString();
         else
         {
            Item old = m_regS1;

            // eventually push parameters if format is required
            int params = 0;
            if( format.size() != 0 )
            {
               pushParameter( new CoreString(format) );
               params = 1;
            }

            // atomically call the item
            callItemAtomic( propString, params );

            m_regS1 = old;
            // if regA is already a string, it's a quite light operation.
            regA().toString( target );
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
   // if we have nowhere to return...
   if( m_stackBase == 0 )
   {
      terminateCurrentContext();
      return;
   }

   // Get the stack frame.
   StackFrame &frame = *(StackFrame *) m_stack->at( m_stackBase - VM_FRAME_SPACE );

   // if the stack frame requires an end handler...
   // ... but only if not unrolling a stack because of error...
   if ( ! hadStoppingEvent() && frame.m_endFrameFunc != 0 )
   {
      // reset pc-next to allow re-call of this frame in case of need.
      m_pc_next = m_pc;
      // if the frame requires to stay here, return immediately
      if ( frame.m_endFrameFunc( this ) )
      {
         return;
      }
   }

   // Ok, we can unroll the stak.
   // reset bidings and self
   m_regBind = frame.m_binding;
   m_regS1 = frame.m_self;

   if( frame.m_break )
   {
      m_event = eventReturn;
   }

   // change symbol
   m_symbol = frame.m_symbol;
   m_pc_next = frame.m_ret_pc;

   // eventually change active module.

   m_currentModule = frame.m_module;
   m_currentGlobals = &m_currentModule->globals();
   if ( m_symbol != 0 && m_symbol->isFunction() )
      m_code = m_symbol->getFuncDef()->code();

   // reset try frame
   m_tryFrame = frame.m_try_base;

   // reset stack base and resize the stack
   uint32 oldBase = m_stackBase -frame.m_param_count - VM_FRAME_SPACE;
   m_stackBase = frame.m_stack_base;
   m_stack->resize( oldBase );

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

   while ( lower < higher - 1 )
   {
      pos = base + point * SEEK_STEP;
      paragon = m_currentModule->module()->getString( endianInt32(*reinterpret_cast< int32 *>( pos )));
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
   paragon = currentModule()->getString( endianInt32(*reinterpret_cast< int32 *>( base + lower * SEEK_STEP )));
   if ( paragon != 0 && *paragon == *value )
   {
      landing =  endianInt32( *reinterpret_cast< uint32 *>( base + lower * SEEK_STEP + sizeof( int32 ) ) );
      return true;
   }

   if ( lower != higher )
   {
      paragon = currentModule()->getString( endianInt32(*reinterpret_cast< int32 *>( base + higher * SEEK_STEP )));
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

   while ( base < target )
   {
      Symbol *sym = currentModule()->getSymbol( endianInt32( *reinterpret_cast< int32 *>( base ) ) );

      fassert( sym );
      if ( sym == 0 )
         return false;

      switch( sym->type() )
      {
         case Symbol::tlocal:
            if( *stackItem( m_stackBase + VM_FRAME_SPACE +  sym->itemId() ).dereference() == *item )
               goto success;
         break;

         case Symbol::tparam:
            if( *param( sym->itemId() ) == *item )
               goto success;
         break;

         default:
            if( *moduleItem( sym->itemId() ).dereference() == *item )
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

   while ( base < target )
   {
      Symbol *sym = currentModule()->getSymbol( endianInt32( *reinterpret_cast< int32 *>( base ) ) );
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
               const CoreObject *obj = itm->asObjectSafe();
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
               if( itm->asObject() == cfr->asObjectSafe() )
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
            if ( itm->isObject() && itm->asObjectSafe()->derivedFrom( *cfr->asString() ) )
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
   Item frame1( (((int64) landingPC) << 32) | (int64) m_tryFrame );
   m_tryFrame = m_stack->size();
   m_stack->push( &frame1 );
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
         module( m_currentModule->name() ) )
      );
      return;
   }

   // get the frame and resize the stack
   int64 tf_land = m_stack->itemAt( m_tryFrame ).asInteger();
   m_stack->resize( m_tryFrame );

   // Change the try frame, and eventually move the PC to the proper position
   m_tryFrame = (uint32) tf_land;
   if( moveTo )
   {
      m_pc_next = (uint32)(tf_land>>32);
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
   return chr >= 'A' || (chr >= '0' && chr <= '9') || chr == '_';
}


Item *VMachine::findLocalSymbolItem( const String &symName ) const
{
   // parse self and sender
   if( symName == "self" )
   {
      return const_cast<Item *>(&self());
   }

   // find the symbol
   const Symbol *sym = currentSymbol();
   if ( sym != 0 )
   {
      // get the relevant symbol table.
      const SymbolTable *symtab;
      switch( sym->type() )
      {
      case Symbol::tclass:
         symtab = &sym->getClassDef()->symtab();
         break;

      case Symbol::tfunc:
         symtab = &sym->getFuncDef()->symtab();
         break;

      case Symbol::textfunc:
         symtab = sym->getExtFuncDef()->parameters();
         break;

      default:
         symtab = 0;
      }
      if ( symtab != 0 )
         sym = symtab->findByName( symName );
      else
         sym = 0; // try again
   }


   // -- not a local symbol? -- try the global module table.
   if( sym == 0 )
   {
      sym = currentModule()->findGlobalSymbol( symName );
      // still zero? Let's try the global symbol table.
      if( sym == 0 )
      {
         Item *itm = findGlobalItem( symName );
         if ( itm != 0 )
            return itm->dereference();
      }
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


bool VMachine::findLocalVariable( const String &name, Item &itm ) const
{
   // item to be returned.
   String sItemName;
   uint32 squareLevel = 0;
   uint32 len = name.length();

   typedef enum {
      initial,
      firstToken,
      interToken,
      dotAccessor,
      dotArrayAccessor,
      dotDictAccessor,
      squareAccessor,
      postSquareAccessor,
      singleQuote,
      doubleQuote,
      strEscapeSingle,
      strEscapeDouble
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
               return false;

            state = firstToken;
            sItemName.append( chr );
         break;

         //===================================================
         // Parse first token. It must be a valid local symbol
         case firstToken:
            if ( vmIsWhiteSpace( chr ) || chr == '.' || chr == '[' )
            {
               Item *lsi = findLocalSymbolItem( sItemName );

               // item not found?
               if( lsi == 0 )
                  return false;

               itm = *lsi;
               // set state accordingly to chr.
               goto resetState;
            }
            else if ( vmIsTokenChr( chr ) )
            {
               sItemName.append( chr );
            }
            else {
               // invalid format
               return false;
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
               // also, notice that we change the item itself.
               Item prop;
               if ( !itm.asObjectSafe()->getProperty( sItemName, prop ) )
                  return false;

               prop.methodize( itm );
               itm = prop;

               // set state accordingly to chr.
               goto resetState;
            }
            else if ( vmIsTokenChr( chr ) )
            {
               sItemName.append( chr );
            }
            else
               return false;
         break;

         case dotArrayAccessor:
            // wating for a complete token.

            if ( vmIsWhiteSpace( chr ) || chr == '.' || chr == '[' )
            {
               // ignore leading ws.
               if( sItemName.size() == 0 && vmIsWhiteSpace( chr ) )
                  break;

               // access the item. We know it's an object or we wouldn't be in this state.
               // also, notice that we change the item itself.
               Item *tmp;
               if ( ( tmp = itm.asArray()->getProperty( sItemName ) ) == 0 )
                  return false;

               if ( tmp->isFunction() )
                  tmp->setMethod( itm, tmp->asFunction() );

               itm = *tmp;
               // set state accordingly to chr.
               goto resetState;
            }
            else if ( vmIsTokenChr( chr ) )
            {
               sItemName.append( chr );
            }
            else
               return false;
         break;

         case dotDictAccessor:
            // wating for a complete token.

            if ( vmIsWhiteSpace( chr ) || chr == '.' || chr == '[' )
            {
               // ignore leading ws.
               if( sItemName.size() == 0 && vmIsWhiteSpace( chr ) )
                  break;

               // access the item. We know it's an object or we wouldn't be in this state.
               // also, notice that we change the item itself.
               Item *tmp;
               if ( ( tmp = itm.asDict()->find( sItemName ) ) == 0 )
                  return false;

               if ( tmp->isFunction() )
                  tmp->setMethod( itm, tmp->asFunction() );
               itm = *tmp;

               // set state accordingly to chr.
               goto resetState;
            }
            else if ( vmIsTokenChr( chr ) )
            {
               sItemName.append( chr );
            }
            else
               return false;
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
                     Item *lsi = parseSquareAccessor( itm, sItemName );
                     if( lsi == 0 )
                        return false;
                     itm = *lsi;

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
                  Item *lsi = parseSquareAccessor( itm, sItemName );
                  if( lsi == 0 )
                     return false;
                  itm = *lsi;

                  goto resetState;
               }
               else
                  sItemName.append( chr );
            }
            else if( ! vmIsWhiteSpace( chr ) )
            {
               return false;
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
                  if( itm.isObject() )
                     state = dotAccessor;
                  else if( itm.isArray() )
                     state = dotArrayAccessor;
                  else if( itm.isDict() && itm.asDict()->isBlessed() )
                     state = dotDictAccessor;
                  else
                     return false;
               break;

               case '[':
                  if( ! itm.isDict() && ! itm.isArray() )
                     return false;

                  state = squareAccessor;
                  squareLevel = 1;
               break;

               default:
                  if( ! vmIsWhiteSpace( chr ) )
                     return false;
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
            if( itm.isObject() )
               state = dotAccessor;
            else if ( itm.isArray() )
               state = dotArrayAccessor;
            else if ( itm.isDict() && itm.asDict()->isBlessed() )
               state = dotDictAccessor;
            else
               return false;
         break;

         case '[':
            if( ! itm.isDict() && ! itm.isArray() )
               return false;

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
      return false;

   // Success
   return true;
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
   String da;

   if( firstChar >= '0' && firstChar <= '9' )
   {
      // try to parse a number.
      int64 num;
      if( accessor.parseInt( num ) )
         acc.setInteger( num );
      else
         return 0;
   }
   else if( firstChar == '\'' || firstChar == '"' )
   {
      // arrays cannot be accessed by strings.
      if( accessed.isArray() )
         return 0;

      da = accessor.subString( 1, accessor.length() - 1 );
      acc.setString( &da );
   }
   else {
      // reparse the accessor as a token
      if( ! findLocalVariable( accessor, acc ) )
         return 0;
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

            default: // compiler warning no-op
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
            uint32 posEnd;
            Item itm;

            if( posColon != String::npos ) {
               posEnd = posColon;
            }
            else if( posPipe != String::npos )
            {
               posEnd = posPipe;
            }
            else {
               posEnd = pos2;
            }

            if ( ! findLocalVariable( src.subString( pos0, posEnd ), itm )  )
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

               if( ! fmt.format( this, *itm.dereference(), temp ) ) {
                  return return_error_parse_fmt;
               }
            }
            // do we have a toString parameter?
            else if( posPipe != String::npos )
            {
               itemToString( temp, &itm, src.subString( posPipe+1, pos2 ) );
            }
            else {
               // otherwise, add the toString version (todo format)
               // append to target.
               itemToString( temp, &itm );
            }

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


void VMachine::referenceItem( Item &target, Item &source )
{
   if( source.isReference() ) {
      target.setReference( source.asReference() );
   }
   else {
      GarbageItem *itm = new GarbageItem( source );
      source.setReference( itm );
      target.setReference( itm );
   }
}


static bool vm_func_eval( VMachine *vm )
{
   CoreArray *arr = vm->local( 0 )->asArray();
   uint32 count = (uint32) vm->local( 1 )->asInteger();

   // interrupt functional sequence request?
   if ( vm->regA().isOob() && vm->regA().isInteger() )
   {
      int64 val =  vm->regA().asInteger();
      if ( val == 1 || val == 0 )
         return false;
   }

   // let's push other function's return value
   if ( vm->regA().isLBind() )
   {
      if ( vm->regA().isFutureBind() )
      {
         vm->regBind().flagsOn( 0xF0 );
      }
      else {
         String *binding = vm->regA().asLBind();
         Item *bind = vm->getBinding( *binding );
         if ( bind == 0 )
         {
            vm->regA().setReference( new GarbageItem( Item() ) );
            vm->setBinding( *binding, vm->regA() );
         }
         else {
            //fassert( bind->isReference() );
            vm->regA() = *bind;
         }
      }
   }

   vm->pushParameter( vm->regA() );

   // fake a call return
   while ( count < arr->length() )
   {
      *vm->local( 1 ) = (int64) count+1;
      if ( vm->functionalEval( arr->at(count) ) )
      {
         return true;
      }
      vm->pushParameter( vm->regA() );
      ++count;
   }

   // done? -- have we to perform a last reduction call?

   if( count > 0 && vm->local( 2 )->isCallable() )
   {
      vm->returnHandler(0);
      vm->callFrame( *vm->local( 2 ), count - 1 );
      return true;
   }

   // if the first element is not callable, generate an array
   CoreArray *array = new CoreArray( count );
   Item *data = array->elements();
   int32 base = vm->currentStack().size() - count;

   for ( uint32 i = 0; i < count; i++ ) {
      data[ i ] = vm->currentStack().itemAt(i + base);
   }
   array->length( count );
   vm->regA() = array;
   vm->currentStack().resize( base );

   return false;
}


bool VMachine::functionalEval( const Item &itm, uint32 paramCount, bool retArray )
{
   // An array
   switch( itm.type() )
   {
      case FLC_ITEM_ARRAY:
      {
         CoreArray *arr = itm.asArray();
         // prepare for parametric evaluation
         fassert( m_stackBase + paramCount <= m_stack->size() );
         for( uint32 pi = 1; pi <= paramCount; ++pi )
         {
            String s;
            s.writeNumber( (int64) pi );
            arr->setProperty(s, m_stack->itemAt( m_stack->size() - pi ) );
         }

         createFrame(0);

         if ( m_regBind.isNil() )
            m_regBind = arr->makeBindings();

         // great. Then recursively evaluate the parameters.
         uint32 count = arr->length();
         if ( count > 0 )
         {
            // if the first element is an ETA function, just call it as frame and return.
            if ( (*arr)[0].isFunction() && (*arr)[0].asFunction()->symbol()->isEta() )
            {
               callFrame( arr, 0 );
               return true;
            }

            // create two locals; we may need it
            addLocals( 2 );
            // time to install our handleres
            returnHandler( vm_func_eval );
            *local(0) = itm;
            *local(1) = (int64)0;

            for ( uint32 l = 0; l < count; l ++ )
            {
               const Item &citem = (*arr)[l];
               *local(1) = (int64)l+1;
               if ( functionalEval( citem ) )
               {
                  return true;
               }

               if ( m_regA.isFutureBind() )
               {
                  // with this marker, the next call operation will search its parameters.
                  // Let's consider this a temporary (but legitimate) hack.
                  m_regBind.flags( 0xF0 );
               }

               pushParameter( m_regA );
            }
            // we got nowere to go
            returnHandler( 0 );

            // is there anything to call? -- is the first element an atom?
            // local 2 is the first element we have pushed
            if( local(2)->isCallable() )
            {
               callFrame( *local(2), count-1 );
               return true;
            }
         }

         // if the first element is not callable, generate an array
         if( retArray )
         {
            CoreArray *array = new CoreArray( count );
            Item *data = array->elements();
            int32 base = m_stack->size() - count;

            for ( uint32 i = 0; i < count; i++ ) {
               data[ i ] = m_stack->itemAt(i + base);
            }
            array->length( count );
            m_regA = array;
         }
         else {
            m_regA.setNil();
         }
         callReturn();
      }
      break;

      case FLC_ITEM_LBIND:
         if ( ! itm.isFutureBind() )
         {
            if ( m_regBind.isDict() )
            {
               Item *bind = getBinding( *itm.asLBind() );
               if ( bind == 0 )
               {
                  m_regA.setReference( new GarbageItem( Item() ) );
                  setBinding( *itm.asLBind(), m_regA );
               }
               else {
                  //fassert( bind->isReference() );
                  m_regA = *bind;
               }
            }
            else
               m_regA.setNil();

            break;
         }
         // fallback

      default:
         m_regA = itm;
   }

   return false;
}

Item *VMachine::findGlobalItem( const String &name ) const
{
   const SymModule *sm = findGlobalSymbol( name );
   if ( sm == 0 ) return 0;
   return sm->item()->dereference();
}


LiveModule *VMachine::findModule( const String &name )
{
   LiveModule **lm =(LiveModule **) m_liveModules.find( &name );
   if ( lm != 0 )
      return *lm;
   return 0;
}


Item *VMachine::findWKI( const String &name ) const
{
   const SymModule *sm = (SymModule *) m_wellKnownSyms.find( &name );
   if ( sm == 0 ) return 0;
   return sm->liveModule()->wkitems().itemPtrAt( sm->wkiid() );
}


bool VMachine::unlink( const Runtime *rt )
{
   for( uint32 iter = 0; iter < rt->moduleVector()->size(); ++iter )
   {
      if (! unlink( rt->moduleVector()->moduleAt( iter ) ) )
         return false;
   }

   return true;
}


bool VMachine::unlink( const Module *module )
{
   MapIterator iter;
   if ( !m_liveModules.find( &module->name(), iter ) )
      return false;

   // get the thing
   LiveModule *lm = *(LiveModule **) iter.currentValue();

   // ensure this module is not the active one
   if ( m_currentModule == lm )
   {
      return false;
   }

   // delete all the exported and well known symbols
   MapIterator stiter = lm->module()->symbolTable().map().begin();
   while( stiter.hasCurrent() )
   {
      Symbol *sym = *(Symbol **) stiter.currentValue();
      if ( sym->isWKS() )
         m_wellKnownSyms.erase( &sym->name() );
      else if ( sym->exported() )
         m_globalSyms.erase( &sym->name() );

      stiter.next();
   }

   // delete the iterator from the map
   m_liveModules.erase( iter );

   //detach the object, so it becomes an invalid callable reference
   lm->detachModule();

   // delete the key, which will detach the module, if found.
   return true;
}


bool VMachine::interrupted( bool raise, bool reset, bool dontCheck )
{
   if( dontCheck || m_systemData.interrupted() )
   {
      if( reset )
         m_systemData.resetInterrupt();

      if ( raise )
      {
         uint32 line = m_symbol->isFunction() ?
            currentModule()->getLineAt( m_symbol->getFuncDef()->basePC() + m_pc )
            : 0;

         raiseError( new InterruptedError(
            ErrorParam( e_interrupted ).origin( e_orig_vm ).
               symbol( m_symbol->name() ).
               module( m_currentModule->name() ).
               line( line )
            ) );
      }

      return true;
   }

   return false;
}

Item *VMachine::getBinding( const String &bind ) const
{
   if ( ! m_regBind.isDict() )
      return 0;

   return m_regBind.asDict()->find( bind );
}

Item *VMachine::getSafeBinding( const String &bind )
{
   if ( ! m_regBind.isDict() )
      return 0;

   Item *found = m_regBind.asDict()->find( bind );
   if ( found == 0 )
   {
      m_regBind.asDict()->insert( new CoreString( bind ), Item() );
      found = m_regBind.asDict()->find( bind );
      found->setReference( new GarbageItem( Item() ) );
   }

   return found;
}


bool VMachine::setBinding( const String &bind, const Item &value )
{
   if ( ! m_regBind.isDict() )
      return false;

   m_regBind.asDict()->insert( new CoreString(bind), value );
   return true;
}


CoreSlot* VMachine::getSlot( const String& slotName, bool create )
{
   m_slot_mtx.lock();

   MapIterator iter;
   if ( ! m_slots.find( &slotName, iter ) )
   {
      if ( create )
      {
         CoreSlot* cs = new CoreSlot( slotName );
         m_slots.insert( &slotName, cs );
         m_slot_mtx.unlock();
         return cs;
      }
      m_slot_mtx.unlock();
      return 0;
   }

   // get the thing
   CoreSlot *cs = *(CoreSlot **) iter.currentValue();
   m_slot_mtx.unlock();
   return cs;
}

void VMachine::removeSlot( const String& slotName )
{
   MapIterator iter;

   m_slot_mtx.lock();
   if ( m_slots.find( &slotName, iter ) )
   {
      m_slots.erase( iter );
      // erase will decrefc, because of item traits in m_slots
   }
   m_slot_mtx.unlock();

}

void VMachine::markSlots( uint32 mark )
{
   MapIterator iter = m_slots.begin();
   m_slot_mtx.lock();
   while( iter.hasCurrent() )
   {
      (*(CoreSlot**) iter.currentValue() )->gcMark( mark );
      iter.next();
   }
   m_slot_mtx.unlock();
}


bool VMachine::consumeSignal()
{
   uint32 base = m_stackBase;

   while( base != 0 )
   {
      StackFrame &frame = *(StackFrame *) m_stack->at( base - VM_FRAME_SPACE );
      if( frame.m_endFrameFunc == coreslot_broadcast_internal )
      {
         frame.m_endFrameFunc = 0;
         // eventually call the onMessageComplete
         Item *msgItem = (Item *) m_stack->at( base + 4 );  // local(4)
         if( msgItem->isInteger() )
         {
            VMMessage* msg = (VMMessage*) msgItem->asInteger();
            msg->onMsgComplete( true );
            delete msg;
         }

         return true;
      }

      base = frame.m_stack_base;
   }

   return false;
}


void VMachine::gcEnable( bool mode )
{
   m_bGcEnabled = mode;
}

bool VMachine::isGcEnabled() const
{
   return m_bGcEnabled;
}


VMContext* VMachine::coPrepare( int32 pSize )
{
   // create a new context
   VMContext *ctx = new VMContext( this );

   // if there are some parameters...
   if ( pSize > 0 )
   {
      // avoid reallocation afterwards.
      ctx->getStack()->reserve( pSize );
      // copy flat
      for( int32 i = 0; i < pSize; i++ )
      {
         ctx->getStack()->push( &m_stack->itemAt( m_stack->size() - pSize + i ) );
      }
      m_stack->resize( m_stack->size() - pSize );
   }
   // rotate the context
   m_contexts.pushBack( ctx );
   putAtSleep( ctx, 0.0 );
   return ctx;
}


bool VMachine::callCoroFrame( const Item &callable, int32 pSize )
{
   if ( ! callable.isCallable() )
      return false;

   // create a new context
   VMContext *ctx = new VMContext( this );

   // if there are some parameters...
   if ( pSize > 0 )
   {
      // avoid reallocation afterwards.
      ctx->getStack()->reserve( pSize );
      // copy flat
      for( int32 i = 0; i < pSize; i++ )
      {
         ctx->getStack()->push( &m_stack->itemAt( m_stack->size() - pSize + i ) );
      }
      m_stack->resize( m_stack->size() - pSize );
   }
   m_contexts.pushBack( ctx );

   // rotate the context
   m_currentContext->save( this );
   putAtSleep( m_currentContext, 0.0 );
   m_currentContext = ctx;
   ctx->restore( this );
   // fake the frame as a pure return value; this will force this coroutine to terminate
   // without peeking any code in the module.
   m_pc = i_pc_call_external_return;
   m_pc_next = i_pc_call_external_return;
   callFrame( callable, pSize );

   return true;

}

//=====================================================================================
// messages
//

void VMachine::postMessage( VMMessage *msg )
{
   // can we post now?

   if ( m_baton.tryAcquire() )
   {
      processMessage( msg );
      execFrame();
      m_baton.release();
   }
   else
   {
      // ok, wa can't do it now; post the message
      m_mtx_mesasges.lock();

      if ( m_msg_head == 0 )
      {
         m_msg_head = msg;
      }
      else {
         m_msg_tail->append( msg );
      }

      // reach the end of the msg list and set the new tail
      while( msg->next() != 0 )
         msg = msg->next();

      m_msg_tail = msg;

      // also, ask for early checks.
      // We're really not concerned about spurious reads here or in the other thread,
      // everything would be ok even without this operation. It's just ok if some of
      // the two threads sync on this asap.
      m_opNextCheck = m_opCount;

      m_mtx_mesasges.unlock();
   }
}

void VMachine::processMessage( VMMessage *msg )
{
   // find the slot
   CoreSlot* slot = getSlot( msg->name(), false );
   if ( slot == 0 || slot->empty() )
   {
      msg->onMsgComplete( false );
      delete msg;
   }

   // create the coroutine
   uint32 pcnext = m_pc_next;
   m_pc_next = i_pc_call_external_return;
   m_pc = i_pc_call_external_return;
   coPrepare(0);
   m_pc_next = pcnext;
   for ( uint32 i = 0; i < msg->paramCount(); ++i )
   {
      pushParameter( *msg->param(i) );
   }

   // create the frame used by the broadcast process
   createFrame( msg->paramCount() );

   // prepare the broadcast in the frame.
   slot->prepareBroadcast( this, 0, msg->paramCount(), msg );
   callReturn();
}

void VMachine::performGC( bool bWaitForCollect )
{
   m_bWaitForCollect = bWaitForCollect;
   memPool->idleVM( this, true );
   m_eGCPerformed.wait();
}

//=====================================================================================
// baton
//

void VMBaton::release()
{
   Baton::release();
   // See if the memPool has anything interesting for us.
   memPool->idleVM( m_owner );
}

void VMBaton::releaseNotIdle()
{
   Baton::release();
}

void VMBaton::onBlockedAcquire()
{
   // See if the memPool has anything interesting for us.
   memPool->idleVM( m_owner );
}


GarbageLock *VMachine::lock( const Item &itm )
{
   GarbageLock *ptr = new GarbageLock( itm );

   m_mtx_lockitem.lock();
   ptr->prev( m_lockRoot );
   ptr->next( m_lockRoot->next() );
   m_lockRoot->next()->prev( ptr );
   m_lockRoot->next( ptr );
   m_mtx_lockitem.unlock();

   return ptr;
}


void VMachine::unlock( GarbageLock *ptr )
{
   fassert( ptr != m_lockRoot );

   // frirst: remove the item from the availability pool
   m_mtx_lockitem.lock();
   ptr->next()->prev( ptr->prev() );
   ptr->prev()->next( ptr->next() );
   m_mtx_lockitem.unlock();

   delete ptr;
}


void VMachine::markLocked()
{
   fassert( m_lockRoot != 0 );
   // do the same for the locked pools
   m_mtx_lockitem.lock();
   GarbageLock *rlock = this->m_lockRoot;
   GarbageLock *lock = rlock;
   do
   {
      memPool->markItem( lock->item() );
      lock = lock->next();
   } while( lock != rlock );
   m_mtx_lockitem.unlock();
}

uint32 VMachine::generation() const
{
   return m_generation;
}


void VMachine::setupScript( int argc, char** argv )
{
   if ( m_mainModule != 0 )
   {
      Item *scriptName = findGlobalItem( "scriptName" );
      if ( scriptName != 0 )
         *scriptName = new CoreString( m_mainModule->module()->name() );

      Item *scriptPath = findGlobalItem( "scriptPath" );
      if ( scriptPath != 0 )
         *scriptPath = new Falcon::CoreString( m_mainModule->module()->name() );
   }

   Falcon::Item *args = findGlobalItem( "args" );
   if ( args != 0 )
   {
      // create the arguments.
      // It is correct to pass an empty array if we haven't any argument to pass.
      Falcon::CoreArray *argsArray = new Falcon::CoreArray;
      for( int i = 0; i < argc; i ++ )
      {
         argsArray->append( new Falcon::CoreString( argv[i] ) );
      }
      *args = argsArray;
   }
}

}

/* end of vm.cpp */
