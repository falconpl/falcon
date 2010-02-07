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
#include <falcon/vmevent.h>
#include <falcon/lineardict.h>

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
   m_refcount(1),
   m_break(false)
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
   m_refcount(1),
   m_break(false)
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
   m_onFinalize = 0;
   m_userData = 0;
   m_bhasStandardStreams = false;
   m_loopsGC = FALCON_VM_DFAULT_CHECK_LOOPS;
   m_loopsContext = FALCON_VM_DFAULT_CHECK_LOOPS;
   m_loopsCallback = 0;
   m_opLimit = 0;
   m_generation = 0;
   m_bSingleStep = false;
   m_stdIn = 0;
   m_stdOut = 0;
   m_stdErr = 0;
   m_launchAtLink = true;
   m_bGcEnabled = true;
   m_bWaitForCollect = false;
   m_bPirorityGC = false;


   resetCounters();

   // this initialization must be performed by all vms.
   m_mainModule = 0;
   m_allowYield = true;

   m_opCount = 0;

   // This vectror has also context ownership -- when we remove a context here, it's dead
   m_contexts.deletor( ContextList_deletor );

   // finally we create the context (and the stack)
   m_currentContext = new VMContext;

   // saving also the first context for accounting reasons.
   m_contexts.pushBack( m_currentContext );

   m_opHandlers = (tOpcodeHandler *) memAlloc( FLC_PCODE_COUNT * sizeof( tOpcodeHandler ) );

   m_metaClasses = (CoreClass**) memAlloc( FLC_ITEM_COUNT * sizeof(CoreClass*) );
   memset( m_metaClasses, 0, FLC_ITEM_COUNT * sizeof(CoreClass*) );

   // Search path
   appSearchPath( Engine::getSearchPath() );

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
   m_opHandlers[ P_LDAS] = opcodeHandler_LDAS;
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
   m_opHandlers[ P_LSB ] = opcodeHandler_LSB;
   m_opHandlers[ P_SELE ] = opcodeHandler_SELE;
   m_opHandlers[ P_INDI ] = opcodeHandler_INDI;
   m_opHandlers[ P_STEX ] = opcodeHandler_STEX;
   m_opHandlers[ P_TRAC ] = opcodeHandler_TRAC;
   m_opHandlers[ P_WRT ] = opcodeHandler_WRT;
   m_opHandlers[ P_STO ] = opcodeHandler_STO;
   m_opHandlers[ P_FORB ] = opcodeHandler_FORB;
   m_opHandlers[ P_EVAL ] = opcodeHandler_EVAL;
   m_opHandlers[ P_CLOS ] = opcodeHandler_CLOS;
   m_opHandlers[ P_PSHL ] = opcodeHandler_PSHL;
   m_opHandlers[ P_OOB ] = opcodeHandler_OOB;
   m_opHandlers[ P_TRDN ] = opcodeHandler_TRDN;
   m_opHandlers[ P_EXEQ ] = opcodeHandler_EXEQ;

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

   if ( m_onFinalize != 0 )
      m_onFinalize( this );

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
}


LiveModule* VMachine::link( Runtime *rt )
{
   fassert(rt);
   // link all the modules in the runtime from first to last.
   // FIFO order is important.
   uint32 listSize = rt->moduleVector()->size();
   LiveModule** lmodList = new LiveModule*[listSize];
   LiveModule* lmod = 0;

   //Make sure we catch falcon errors to delete lmodList afterwards.
   try {
      uint32 iter;
      for( iter = 0; iter < listSize; ++iter )
      {
         ModuleDep *md = rt->moduleVector()->moduleDepAt( iter );
         if ( (lmod = prelink( md->module(),
                          rt->hasMainModule() && (iter + 1 == listSize),
                          md->isPrivate() ) ) == 0
         )
         {
            delete [] lmodList;
            return 0;
         }

         // use the temporary storage.
         lmodList[ iter ] = lmod;
      }

      // now again, do the complete phase.
      for( iter = 0; iter < listSize; ++iter )
      {
          if ( ! completeModLink( lmodList[ iter ] ) )
          {
             delete [] lmodList;
             return 0;
          }
      }

      // returns the topmost livemodule
      delete [] lmodList;
   }
   catch( Error *err )
   {
      delete [] lmodList;
      throw err;
   }
   return lmod;
}


LiveModule *VMachine::link( Module *mod, bool isMainModule, bool bPrivate )
{
   // Ok, the module is now in.
   // We can now increment reference count and add it to ourselves
   LiveModule *livemod = prelink( mod, isMainModule, bPrivate );

   if ( livemod && completeModLink( livemod ) )
      return livemod;

   return 0;
}

LiveModule *VMachine::prelink( Module *mod, bool isMainModule, bool bPrivate )
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
         oldMod->setPrivate( false );
      }

      return oldMod;
   }

   // Ok, the module is now in.
   // We can now increment reference count and add it to ourselves
   LiveModule *livemod = new LiveModule( mod, bPrivate );

   // set this as the main module if required.
   if ( isMainModule )
      m_mainModule = livemod;

   // no need to free on failure: livemod are garbaged
   livemod->mark( generation() );

   // then we always need the symbol table.
   const SymbolTable *symtab = &livemod->module()->symbolTable();

   // A shortcut
   ItemArray& globs = livemod->globals();

   // resize() creates a series of NIL items.
   globs.resize( symtab->size()+1 );

   bool success = true;
   // now, the symbol table must be traversed.
   MapIterator iter = symtab->map().begin();
   while( iter.hasCurrent() )
   {
      Symbol *sym = *(Symbol **) iter.currentValue();

      if ( ! sym->isUndefined() )
      {
         if ( ! linkDefinedSymbol( sym, livemod ) )
         {
            // but continue to expose other errors as well.
            success = false;
         }
      }

      // next symbol
      iter.next();
   }

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

   return livemod;
}


bool VMachine::completeModLink( LiveModule *livemod )
{
   // we need to record the classes in the module as they have to be evaluated last.
   SymbolList modClasses;
   SymbolList modObjects;

   // then we always need the symbol table.
   const SymbolTable *symtab = &livemod->module()->symbolTable();

   // we won't be preemptible during link
   bool atomic = m_currentContext->atomicMode();
   m_currentContext->atomicMode(true);

   bool success = true;
   // now, the symbol table must be traversed.
   MapIterator iter = symtab->map().begin();
   while( iter.hasCurrent() )
   {
      Symbol *sym = *(Symbol **) iter.currentValue();
      if ( sym->isUndefined() )
      {
         if (! linkUndefinedSymbol( sym, livemod ) )
            success = false;
      }
      else
      {
         // save classes and objects for later linking.
         if( sym->type() == Symbol::tclass )
            modClasses.pushBack( sym );
         else if ( sym->type() == Symbol::tinst )
            modObjects.pushBack( sym );
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

   if ( success )
   {
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

   }

   // Initializations of module objects is complete; return to non-atomic mode
   m_currentContext->atomicMode( atomic );

   // return zero and dispose of the module if not succesful.
   if ( ! success )
   {
      // LiveModule is garbageable, cannot be destroyed.
      return false;
   }

   // and for last, export all the services.
   MapIterator svmap_iter = livemod->module()->getServiceMap().begin();
   while( svmap_iter.hasCurrent() )
   {
      // throws on error.
      publishService( *(Service ** ) svmap_iter.currentValue() );
      svmap_iter.next();
   }


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
   if ( sym->isUndefined() )
   {
      return linkUndefinedSymbol( sym, livemod );
   }
   return  linkDefinedSymbol( sym, livemod );
}


bool VMachine::linkDefinedSymbol( const Symbol *sym, LiveModule *livemod )
{
   // A shortcut
   ItemArray& globs = livemod->globals();

   // Ok, the symbol is defined here. Link (record) it.

   // create an appropriate item here.
   // NOTE: Classes and instances are handled separately.
   switch( sym->type() )
   {
      case Symbol::tfunc:
      case Symbol::textfunc:
         globs[ sym->itemId() ].setFunction( new CoreFunc( sym, livemod ) );
      break;

      case Symbol::tvar:
      case Symbol::tconst:
      {
         Item& itm = globs[ sym->itemId() ];
         VarDef *vd = sym->getVarDef();
         switch( vd->type() ) {
            case VarDef::t_bool: itm.setBoolean( vd->asBool() ); break;
            case VarDef::t_int: itm.setInteger( vd->asInteger() ); break;
            case VarDef::t_num: itm.setNumeric( vd->asNumeric() ); break;
            case VarDef::t_string:
            {
               itm.setString( new CoreString( *vd->asString() ) );
            }
            break;

            default:
               break;
         }
      }
      break;

      // nil when we don't know what it is.
      default:
         globs[ sym->itemId() ].setNil();
   }

   // see if the symbol needs exportation and eventually do that.
   if ( ! exportSymbol( sym, livemod ) )
      return false;

   return true;
}


bool VMachine::linkUndefinedSymbol( const Symbol *sym, LiveModule *livemod )
{
   // A shortcut
   ItemArray& globs = livemod->globals();
   const Module *mod = livemod->module();

   // is the symbol name-spaced?
   uint32 dotPos;

   String localSymName;
   ModuleDepData *depData;
   LiveModule *lmod = 0;

   if ( ( dotPos = sym->name().rfind( "." ) ) != String::npos && sym->imported() )
   {
      String nameSpace = sym->name().subString( 0, dotPos );
      // get the module name for the given module
      depData = mod->dependencies().findModule( nameSpace );
      // if we linked it, it must exist
      // -- but in some cases, the compiler may generate a dotted symbol loaded from external sources
      // -- this is usually an error, so let the undefined error to be declared.
      if ( depData != 0 )
      {
         // ... then find the module in the item
         lmod = findModule( Module::absoluteName(
               depData->isFile() ? nameSpace: depData->moduleName(),
               mod->name() ));

         // we must convert the name if it contains self or if it starts with "."
         if ( lmod != 0 )
            localSymName = sym->name().subString( dotPos + 1 );
      }
   }
   else if ( sym->isImportAlias() )
   {
      depData = mod->dependencies().findModule( sym->getImportAlias()->origModule() );
      // if we linked it, it must exist
      fassert( depData != 0 );

      // ... then find the module in the item
      lmod = findModule( Module::absoluteName(
            depData->moduleName(), mod->name() ));

      if( lmod != 0 )
         localSymName = sym->getImportAlias()->name();
   }

   // If we found it it...
   if ( lmod != 0 )
   {
      Symbol *localSym = lmod->module()->findGlobalSymbol( localSymName );

      if ( localSym != 0 )
      {
         referenceItem( globs[ sym->itemId() ],
            lmod->globals()[ localSym->itemId() ] );
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
            if ( newsym->itemId() >= lmod->globals().length() )
            {
               lmod->globals().resize( newsym->itemId() );
            }

            // now we have space to link it.
            if ( linkCompleteSymbol( newsym, lmod ) )
            {
               referenceItem( globs[ sym->itemId() ], lmod->globals()[newsym->itemId()] );
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
         referenceItem( globs[ sym->itemId() ], *sm->item() );
         return true;
      }
   }

   // try to dynamically load the symbol from flexy modules.
   SymModule symmod;
   if ( linkSymbolDynamic( sym->name(), symmod ) )
   {
      referenceItem( globs[ sym->itemId() ], *symmod.item() );
      return true;
   }

   // We failed every try; raise undefined symbol.
   throw new CodeError(
         ErrorParam( e_undef_sym, sym->declaredAt() ).origin( e_orig_vm ).
         module( mod->name() ).
         extra( sym->name() )
         );
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
   ItemArray& globs = livemod->globals();
   const Module *mod = livemod->module();

      // Is this symbol exported?
   if ( ! livemod->isPrivate() && sym->exported() && sym->name().getCharAt(0) != '_' )
   {
      // as long as the module is referenced, the symbols are alive, and as we
      // hold a reference to the module, we are sure that symbols are alive here.
      // also, in case an entry already exists, the previous item is just overwritten.

      if ( m_globalSyms.find( &sym->name() ) != 0 )
      {
         throw new CodeError( ErrorParam( e_already_def, sym->declaredAt() ).origin( e_orig_vm ).
                  module( mod->name() ).
                  symbol( sym->name() ) );
      }

      SymModule tmp( &globs[ sym->itemId() ], livemod, sym );
      m_globalSyms.insert( &sym->name(), &tmp );

      // export also the instance, if it is not already exported.
      if ( sym->isInstance() )
      {
         Symbol* tsym = sym->getInstance();
         if ( ! tsym->exported() ) {
            SymModule tmp( &globs[ tsym->itemId() ], livemod, tsym );
            m_globalSyms.insert( &tsym->name(), &tmp );
         }
      }
   }

   // Is this symbol a well known item?
   if ( sym->isWKS() )
   {
      if ( m_wellKnownSyms.find( &sym->name() ) != 0 )
      {
         throw
            new CodeError( ErrorParam( e_already_def, sym->declaredAt() ).origin( e_orig_vm ).
                  module( mod->name() ).
                  symbol( sym->name() ).
                  extra( "Well Known Item" )
            );
      }

      SymModule tmp( livemod->wkitems().length(), livemod, sym );
      m_wellKnownSyms.insert( &sym->name(), &tmp );

      // and don't forget to add a copy of the item
      livemod->wkitems().append( globs[ sym->itemId() ] );
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
            if ( newsym->itemId() >= lmod->globals().length() )
            {
               lmod->globals().resize( newsym->itemId() );
            }

            // now we have space to link it.
            if ( linkCompleteSymbol( newsym, lmod ) )
            {
               symdata = SymModule( &lmod->globals()[ newsym->itemId() ], lmod, newsym );
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
   ItemArray& globs = livemod->globals();

   CoreClass *cc = linkClass( livemod, sym );
   if ( cc == 0 )
      return false;

   // we need to add it anyhow to the GC to provoke its destruction at VM end.
   // and hey, you could always destroy symbols if your mood is so from falcon ;-)
   // dereference as other classes may have referenced this item1
   globs[ cc->symbol()->itemId() ].dereference()->setClass( cc );

   // if this class was a WKI, we must also set the relevant exported symbol
   if ( sym->isWKS() )
   {
      SymModule *tmp = (SymModule *) m_wellKnownSyms.find( &sym->name() );
      fassert( tmp != 0 ); // we just added it
      tmp->liveModule()->wkitems()[ tmp->wkiid() ] = cc;
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
   ItemArray& globs = livemod->globals();
   Symbol *cls = obj->getInstance();
   Item *clsItem = globs[ cls->itemId() ].dereference();

   if ( clsItem == 0 || ! clsItem->isClass() ) {
         new CodeError( ErrorParam( e_no_cls_inst, obj->declaredAt() ).origin( e_orig_vm ).
            symbol( obj->name() ).
            module( obj->module()->name() )
      );
      return false;
   }
   else {
      CoreObject *co = clsItem->asClass()->createInstance();
      globs[ obj->itemId() ].dereference()->setObject( co );

      // if this class was a WKI, we must also set the relevant exported symbol
      if ( obj->isWKS() )
      {
         SymModule *tmp = (SymModule *) m_wellKnownSyms.find( &obj->name() );
         fassert ( tmp != 0 );
         tmp->liveModule()->wkitems()[ tmp->wkiid() ] = co;
      }
   }

   return true;
}


void VMachine::initializeInstance( const Symbol *obj, LiveModule *livemod )
{
   ItemArray& globs = livemod->globals();

   Symbol *cls = obj->getInstance();
   if ( cls->getClassDef()->constructor() != 0 )
   {
      SafeItem ctor = *globs[cls->getClassDef()->constructor()->itemId() ].dereference();
      ctor.methodize( *globs[ obj->itemId() ].dereference() );

      // If we can't call, we have a wrong init.
      callItemAtomic( ctor, 0 );
   }

   CoreObject* cobj = globs[ obj->itemId() ].dereference()->asObject();
   if( cobj->generator()->initState() != 0 )
   {
      cobj->setState( "init", cobj->generator()->initState() );
      // If we can't call, we have a wrong init.

      Item enterItem;
      if( cobj->getMethod("__enter", enterItem ) )
      {
         pushParam(Item());
         callItemAtomic( enterItem, 1 );
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
   bool bHasSetGet = false;

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

         // we must keep this information for later.
         if ( e.m_eReflectMode == e_reflectSetGet )
            bHasSetGet = true;
         else
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
            referenceItem( e.m_value, vdmod->lmod->globals()[ sym->itemId() ] );
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

   // complete setter/getter
   if( bHasSetGet )
   {
      for( uint32 pp = 0; pp < table->added(); ++pp )
      {
         PropEntry& pe = table->getEntry(pp);
         if( pe.m_eReflectMode == e_reflectSetGet )
         {
            uint32 pos;

            if( table->findKey(String("__set_") + *pe.m_name, pos ) )
            {
               pe.m_reflection.gs.m_setterId = pos;
            }
            else
            {
               pe.m_reflection.gs.m_setterId = PropEntry::NO_OFFSET;
               pe.m_bReadOnly = true;
            }

            if( table->findKey(String("__get_") + *pe.m_name, pos ) )
            {
               pe.m_reflection.gs.m_getterId = pos;
            }
            else
            {
               pe.m_reflection.gs.m_getterId = PropEntry::NO_OFFSET;
            }
         }
      }
   }

   table->checkProperties();

   return table;
}


CoreClass *VMachine::linkClass( LiveModule *lmod, const Symbol *clssym )
{
   Map props( &traits::t_stringptr(), &traits::t_voidp() );
   Map states( &traits::t_stringptr(), &traits::t_voidp() );

   ObjectFactory factory = 0;
   if( ! linkSubClass( lmod, clssym, props, states, &factory ) )
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

   // apply the state map
   if( ! states.empty() )
   {
      MapIterator siter = states.begin();
      ItemDict* dict = new LinearDict;
      ItemDict* initState = 0;

      while ( siter.hasCurrent() )
      {
         const String* sname = *(String**) siter.currentKey();
         const Map* sd = *(Map**) siter.currentValue();
         ItemDict *sdict = new LinearDict(sd->size());
         MapIterator fiter = sd->begin();

         while( fiter.hasCurrent() )
         {
            const String* fname = *(String**) fiter.currentKey();
            CoreFunc* sfunc = *(CoreFunc**) fiter.currentValue();

            sdict->put(
                  new CoreString( *fname ), sfunc );

            fiter.next();
         }
         delete sd;

         // TODO: See if we can use the const String* form sd->name() here
         dict->put( new CoreString( *sname ), new CoreDict(sdict) );
         if( *sname == "init" )
            initState = sdict;

         siter.next();
      }

      cc->states( dict, initState );
   }


   // ok, now determine the default object factory, if not provided.
   if( factory != 0 )
   {
      cc->factory( factory );
   }
   else
   {
      // use one of our standard factories.
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
      Map &props, Map &states, ObjectFactory *factory )
{
   // first sub-instantiates all the inheritances.
   ClassDef *cd = clssym->getClassDef();
   ListElement *from_iter = cd->inheritance().begin();
   const Module *class_module = clssym->module();

   // If the class is final, we're doomed, as this is called on subclasses
   if( cd->isFinal() )
   {
      throw new CodeError( ErrorParam( e_final_inherit, clssym->declaredAt() ).origin( e_orig_vm ).
            symbol( clssym->name() ).
            module( class_module->name() ) );
   }

   if( *factory != 0 && cd->factory() != 0 )
   {
      // raise an error for duplicated object manager.
      throw new CodeError( ErrorParam( e_inv_inherit2, clssym->declaredAt() ).origin( e_orig_vm ).
            symbol( clssym->name() ).
            module( class_module->name() )
            );
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
         // do we have some circular inheritance
         if ( parent->getClassDef()->checkCircularInheritance( clssym ) )
             throw new CodeError( ErrorParam( e_circular_inh, __LINE__ )
                   .origin( e_orig_vm )
                   .extra( clssym->name() ) );

         // we create the item anew instead of relying on the already linked item.
         if ( ! linkSubClass( lmod, parent, props, states, &subFactory ) )
            return false;
      }
      else if ( parent->isUndefined() )
      {
         // we have already linked the symbol for sure.
         Item *icls = lmod->globals()[ parent->itemId() ].dereference();

         if ( ! icls->isClass() )
         {
            throw new CodeError( ErrorParam( e_inv_inherit, clssym->declaredAt() ).origin( e_orig_vm ).
                  symbol( clssym->name() ).
                  module( class_module->name() )
                  );
         }

         parent = icls->asClass()->symbol();

         if ( parent->getClassDef()->checkCircularInheritance( clssym ) )
          throw new CodeError( ErrorParam( e_circular_inh, __LINE__ )
                .origin( e_orig_vm )
                .extra( clssym->name() ) );

         LiveModule *parmod = findModule( parent->module()->name() );
         if ( ! linkSubClass( parmod, parent, props, states, &subFactory ) )
            return false;
      }
      else
      {
         throw new CodeError( ErrorParam( e_inv_inherit, clssym->declaredAt() ).origin( e_orig_vm ).
                  symbol( clssym->name() ).
                  module( class_module->name() ) );
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

   // and the same for the states.
   MapIterator siter = cd->states().begin();
   while( siter.hasCurrent() )
   {
      String *stateName = *(String **) siter.currentKey();
      StateDef* sd = *(StateDef **) siter.currentValue();

      Map* sfuncs = new Map( &traits::t_stringptr(), &traits::t_voidp() );

      MapIterator fiter = sd->functions().begin();
      while( fiter.hasCurrent() )
      {
         const String* fname = *(String**) fiter.currentKey();
         const Symbol* fsym = *(Symbol**) fiter.currentValue();
         CoreFunc* sfunc = new CoreFunc( fsym, lmod );

         CoreFunc **oldFunc = (CoreFunc **) sfuncs->find( fname );
         if ( oldFunc != 0 )
         {
            delete *oldFunc;
            *oldFunc = sfunc;
         }
         else {
            sfuncs->insert( fname, sfunc );
         }

         fiter.next();
      }

      //==========================
      Map** oldFuncs =(Map**) states.find( stateName );
      if( oldFuncs != 0 )
      {
         delete *oldFuncs;
         *oldFuncs = sfuncs;
      }
      else
      {
         states.insert( stateName, sfuncs );
      }

      siter.next();
   }

   return true;
}


void VMachine::reset()
{
   // first, the trivial resets.

   // reset counters
   resetCounters();

   // clear the accounting of sleeping contexts.
   m_sleepingContexts.clear();

   if ( m_contexts.size() > 1 )
   {
      // clear the contexts
      m_contexts.clear();

      // as our frame, stack and tryframe were in one of the contexts,
      // they have been destroyed.
      m_currentContext = new VMContext;

      // saving also the first context for accounting reasons.
      m_contexts.pushBack( m_currentContext );
   }
   else
   {
      m_currentContext->resetFrames();
   }
}

const SymModule *VMachine::findGlobalSymbol( const String &name ) const
{
   return (SymModule *) m_globalSyms.find( &name );
}


bool VMachine::getCaller( const Symbol *&sym, const Module *&module)
{
   StackFrame* frame = currentFrame();

   if( frame == 0 || frame->m_module == 0 )
      return false;

   sym = frame->m_symbol;
   module = frame->m_module->module();
   return sym != 0 && module != 0;
}

bool VMachine::getCallerItem( Item &caller, uint32 level )
{
   StackFrame* frame = currentFrame();

   while( frame != 0 || level > 0 )
   {
      frame = frame->prev();
      level--;
   }

   if ( frame == 0 || frame->m_module == 0 )
      return false;

   const Symbol* sym = frame->m_symbol;
   const LiveModule* module = frame->m_module;

   caller = module->globals()[ sym->itemId() ];
   if ( ! caller.isFunction() )
      return false;

   // was it a method ?
   if ( ! frame->m_self.isNil() )
   {
      caller.methodize( frame->m_self );
   }

   return true;
}

void VMachine::fillErrorContext( Error *err, bool filltb )
{
   if( currentSymbol() != 0 )
   {
      if ( err->module().size() == 0 )
         err->module( currentModule()->name() );

      if ( err->symbol().size() == 0 )
         err->symbol( currentSymbol()->name() );

      if( currentSymbol()->isFunction() )
         err->line( currentModule()->getLineAt( currentSymbol()->getFuncDef()->basePC() + programCounter() ) );

      err->pcounter( programCounter() );
   }

    if ( filltb )
      fillErrorTraceback( *err );

}

void VMachine::callFrameNow( ext_func_frame_t callbackFunc )
{
   currentFrame()->m_endFrameFunc = callbackFunc;
   switch( m_currentContext->pc() )
   {
      case i_pc_call_external_ctor:
         m_currentContext->pc_next() = i_pc_call_external_ctor_return;
         break;
      case i_pc_call_external:
         m_currentContext->pc_next() = i_pc_call_external_return;
         break;
      default:
         m_currentContext->pc_next() = m_currentContext->pc();
   }
}


void VMachine::callItemAtomic(const Item &callable, int32 paramCount )
{
   bool oldAtomic = m_currentContext->atomicMode();
   m_currentContext->atomicMode( true );
   callFrame( callable, paramCount );
   execFrame();
   m_currentContext->atomicMode( oldAtomic );
}


void VMachine::yield( numeric secs )
{
   if ( m_currentContext->atomicMode() )
   {
      throw new InterruptedError( ErrorParam( e_wait_in_atomic, __LINE__ )
            .origin( e_orig_vm )
            .symbol( "yield" )
            .module( "core.vm" )
            .hard() );
   }

   // be sure to allow yelding.
   m_allowYield = true;

   // a pure sleep time can never be < 0.
   if( secs < 0.0 )
      secs = 0.0;

   m_currentContext->scheduleAfter( secs );

   rotateContext();
}


void VMachine::putAtSleep( VMContext *ctx )
{
   // consider the special case of a context not willing to be awaken
   if( ctx->isWaitingForever() )
   {
      m_sleepingContexts.pushBack( ctx );
      return;
   }

   ListElement *iter = m_sleepingContexts.begin();
   while( iter != 0 ) {
      VMContext *curctx = (VMContext *) iter->data();
      if ( ctx->schedule() < curctx->schedule() || curctx->schedule() < 0.0 ) {
         m_sleepingContexts.insertBefore( iter, ctx );
         return;
      }
      iter = iter->next();
   }

   // can't find it anywhere?
   m_sleepingContexts.pushBack( ctx );
}


void VMachine::reschedule( VMContext *ctx )
{
   ListElement *iter = m_sleepingContexts.begin();

   bool bPlaced = false;
   bool bRemoved = false;
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
         // -- did we find the position where to place it, we did it all.
         // -- go away.
         if ( bPlaced )
            return;

         // but if item was not placed because it has an endless sleep, we have
         // no gain in scanning more than this. So just place at the end
         // (by breaking)
         if( ctx->schedule() < 0.0 )
            break;

         // otherwise, continue working
         bRemoved = true;
         continue;
      }

      // avoid to place twice
      if ( ! bPlaced &&
         ( ctx->schedule() < curctx->schedule() || curctx->schedule() < 0.0 )
         )
      {
         m_sleepingContexts.insertBefore( iter, ctx );
         // if we have also already removed the previous position, we did all
         if( bRemoved )
            return;

         bPlaced = true;
      }

      iter = iter->next();
   }

   // can't find any place to store it?
   if ( ! bPlaced )
      m_sleepingContexts.pushBack( ctx );
}


void VMachine::rotateContext()
{
   putAtSleep( m_currentContext );
   electContext();
}


void VMachine::electContext()
{
   // if there is some sleeping context...
   if ( ! m_sleepingContexts.empty() )
   {
      VMContext *elect = (VMContext *) m_sleepingContexts.front();
      m_sleepingContexts.popFront();

      numeric tgtTime = elect->schedule();

      // changhe the context to the first ready to run.
      m_currentContext = elect;

      // we must move to the next instruction after the context was swapped.
      m_currentContext->pc() = m_currentContext->pc_next();

      // tell the context that it is not waiting anymore, if it was.
      elect->wakeup( false );

      // Is the most ready context willing to sleep?
      if( tgtTime < 0.0 ||  (tgtTime -= Sys::_seconds()) > 0.0 )
      {
         // raise an interrupted error if interrutped.
         // also, this may wait forever (with a tgtime < 0);
         // in this case the function may throw a deadlock error
         onIdleTime( tgtTime );
      }

      m_opCount = 0;
   }
}


void VMachine::terminateCurrentContext()
{
   // don't destroy this context if it's the last one.
   // inspectors outside this VM may want to check it.
   if ( ! m_contexts.empty() && m_contexts.begin()->next() != 0 )
   {
      // scan the contexts and remove the current one.
      ListElement *iter = m_contexts.begin();
      while( iter != 0 ) {
         if( iter->data() == m_currentContext ) {
            m_contexts.erase( iter );
            m_currentContext = 0;
            break;
         }
         iter = iter->next();
      }

      // there must be something sleeping
      fassert( !  m_sleepingContexts.empty() );
      electContext();
   }
   else {
      // we're done
      throw VMEventQuit();
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
            Item old = self();

            // eventually push parameters if format is required
            int params = 0;
            if( format.size() != 0 )
            {
               pushParam( new CoreString(format) );
               params = 1;
            }

            // atomically call the item
            callItemAtomic( propString, params );

            self() = old;
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

      if ( loadInt64( pos ) == value )
      {
         landing = *reinterpret_cast< uint32 *>( pos + sizeof(int64) );
         return true;
      }

      if ( value > loadInt64( pos ) )
         lower = point;
      else
         higher = point;
      point = ( lower + higher ) / 2;
   }

   // see if it was in the last loop
   if ( loadInt64( base + lower * SEEK_STEP ) == value )
   {
      landing =  *reinterpret_cast< uint32 *>( base + lower * SEEK_STEP + sizeof( int64 ) );
      return true;
   }

   if ( lower != higher && loadInt64( base + higher * SEEK_STEP ) == value )
   {
      // YATTA, we found it at last
      landing =  *reinterpret_cast< uint32 *>( base + higher * SEEK_STEP + sizeof( int64 ) );
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

      if ( *reinterpret_cast< int32 *>( pos ) <= value &&
                *reinterpret_cast< int32 *>( pos + sizeof( int32 ) ) >= value)
      {
         landing = *reinterpret_cast< int32 *>( pos + sizeof( int32 ) + sizeof( int32 ) );
         return true;
      }

      if ( value > *reinterpret_cast< int32 *>( pos ) )
         lower = point;
      else
         higher = point;
      point = ( lower + higher ) / 2;
   }

   // see if it was in the last loop
   pos = base + lower * SEEK_STEP;
   if ( *reinterpret_cast< int32 *>( pos ) <= value &&
        *reinterpret_cast< int32 *>( pos + sizeof( int32 ) ) >= value )
   {
      landing =  *reinterpret_cast< uint32 *>( pos + sizeof( int32 ) + sizeof( int32 ) );
      return true;
   }

   if( lower != higher )
   {
      pos = base + higher * SEEK_STEP;
      if ( *reinterpret_cast< int32 *>( pos ) <= value &&
           *reinterpret_cast< int32 *>( pos + sizeof( int32 ) ) >= value )
      {
         // YATTA, we found it at last
         landing = *reinterpret_cast< uint32 *>( pos + sizeof( int32 ) + sizeof( int32 ) );
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
      paragon = currentModule()->getString( *reinterpret_cast< int32 *>( pos ) );
      fassert( paragon != 0 );
      if ( paragon == 0 )
         return false;
      if ( *paragon == *value )
      {
         landing = *reinterpret_cast< int32 *>( pos + sizeof(int32) );
         return true;
      }

      if ( *value > *paragon )
         lower = point;
      else
         higher = point;
      point = ( lower + higher ) / 2;
   }

   // see if it was in the last loop
   paragon = currentModule()->getString( *reinterpret_cast< uint32 *>( base + lower * SEEK_STEP ) );
   if ( paragon != 0 && *paragon == *value )
   {
      landing = *reinterpret_cast< uint32 *>( base + lower * SEEK_STEP + sizeof( int32 ) );
      return true;
   }

   if ( lower != higher )
   {
      paragon = currentModule()->getString( *reinterpret_cast< uint32 *>( base + higher * SEEK_STEP ) );
      if ( paragon != 0 && *paragon == *value )
      {
         // YATTA, we found it at last
         landing = *reinterpret_cast< uint32 *>( base + higher * SEEK_STEP + sizeof( int32 ) );
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
      Symbol *sym = currentModule()->getSymbol( *reinterpret_cast< int32 *>( base ) );

      fassert( sym );
      if ( sym == 0 )
         return false;

      switch( sym->type() )
      {
         case Symbol::tlocal:
            if( *local( sym->itemId() )->dereference() == *item )
               goto success;
         break;

         case Symbol::tparam:
            if( *param( sym->itemId() )->dereference() == *item )
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
   landing = *reinterpret_cast< uint32 *>( base + sizeof(int32) );
   return true;
}

bool VMachine::seekItemClass( const Item *itm, byte *base, uint16 size, uint32 &landing ) const
{
   #undef SEEK_STEP
   #define SEEK_STEP (sizeof(int32) + sizeof(int32))

   byte *target = base + size *SEEK_STEP;

   while ( base < target )
   {
      Symbol *sym = currentModule()->getSymbol( *reinterpret_cast< uint32 *>( base ) );
      fassert( sym );
      if ( sym == 0 )
         return false;

      const Item *cfr;

      if ( sym->isLocal() )
      {
         cfr = local( sym->itemId() )->dereference();
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
   landing = *reinterpret_cast< uint32 *>( base + sizeof(int32) );
   return true;
}

void VMachine::publishService( Service *svr )
{
   Service **srv = (Service **) m_services.find( &svr->getServiceName() );
   if ( srv == 0 )
   {
      m_services.insert( &svr->getServiceName(), svr );
   }
   else {
      throw new CodeError(
            ErrorParam( e_service_adef ).origin( e_orig_vm ).
            extra( svr->getServiceName() ).
            symbol( "publishService" ).
            module( "core.vm" ) );
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

               // access the item. We know it's an object or class or we wouldn't be in this state.
               // also, notice that we change the item itself.
               Item prop;
			   if( itm.isClass() )
			   {
				   const Falcon::Item* requested = itm.asClass()->properties().getValue( sItemName );
				   if( requested == 0 )
					   return false;
				   prop = *requested;
			   }
               else if ( !itm.asObjectSafe()->getProperty( sItemName, prop ) )
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
			if( itm.isObject() || itm.isClass() )
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

   vm->pushParam( vm->regA() );

   // fake a call return
   while ( count < arr->length() )
   {
      *vm->local( 1 ) = (int64) count+1;
      if ( vm->functionalEval( arr->at(count) ) )
      {
         return true;
      }
      vm->pushParam( vm->regA() );
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
   Item *data = array->items().elements();
   int32 base = vm->stack().length() - count;

   memcpy( data, &vm->stack()[base],array->items().esize( count ) );

   array->length( count );
   vm->regA() = array;
   vm->stack().resize( base );

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
         for( uint32 pi = 1; pi <= paramCount; ++pi )
         {
            String s;
            s.writeNumber( (int64) pi );
            arr->setProperty(s, stack()[ stack().length() - pi] );
         }

         createFrame(0);

         if ( regBind().isNil() )
            regBind() = arr->makeBindings();

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

               if ( regA().isFutureBind() )
               {
                  // with this marker, the next call operation will search its parameters.
                  // Let's consider this a temporary (but legitimate) hack.
                  regBind().flags( 0xF0 );
               }

               pushParam( regA() );
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
            Item *data = array->items().elements();
            int32 base = stack().length() - count;

            memcpy( data, &stack()[base], array->items().esize( count ) );
            array->length( count );
            regA() = array;
         }
         else {
            regA().setNil();
         }
         callReturn();
      }
      break;

      case FLC_ITEM_LBIND:
         if ( ! itm.isFutureBind() )
         {
            if ( regBind().isDict() )
            {
               Item *bind = getBinding( *itm.asLBind() );
               if ( bind == 0 )
               {
                  regA().setReference( new GarbageItem( Item() ) );
                  setBinding( *itm.asLBind(), regA() );
               }
               else {
                  //fassert( bind->isReference() );
                  regA() = *bind;
               }
            }
            else
               regA().setNil();

            break;
         }
         // fallback

      default:
         regA() = itm;
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
   return &sm->liveModule()->wkitems()[ sm->wkiid() ];
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
   if ( currentLiveModule() == lm )
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
         uint32 line = currentSymbol()->isFunction() ?
            currentModule()->getLineAt( currentSymbol()->getFuncDef()->basePC() + programCounter() )
            : 0;

         throw new InterruptedError(
            ErrorParam( e_interrupted ).origin( e_orig_vm ).
               symbol( currentSymbol()->name() ).
               module( currentModule()->name() ).
               line( line ) );
      }

      return true;
   }

   return false;
}

Item *VMachine::getBinding( const String &bind ) const
{
   if ( ! regBind().isDict() )
      return 0;

   return regBind().asDict()->find( bind );
}

Item *VMachine::getSafeBinding( const String &bind )
{
   if ( ! regBind().isDict() )
      return 0;

   Item *found = regBind().asDict()->find( bind );
   if ( found == 0 )
   {
      regBind().asDict()->put( new CoreString( bind ), Item() );
      found = regBind().asDict()->find( bind );
      found->setReference( new GarbageItem( Item() ) );
   }

   return found;
}


bool VMachine::setBinding( const String &bind, const Item &value )
{
   if ( ! regBind().isDict() )
      return false;

   regBind().asDict()->put( new CoreString(bind), value );
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
   StackFrame* frame = currentFrame();

   while( frame != 0 )
   {
      if( frame->m_endFrameFunc == coreslot_broadcast_internal )
      {
         frame->m_endFrameFunc = 0;
         const Item& msgItem = frame->localItem(4); // local(4)
         if( msgItem.isInteger() )
         {
            VMMessage* msg = (VMMessage*) msgItem.asInteger();
            msg->onMsgComplete( true );
            delete msg;
         }

         return true;
      }

      frame = frame->prev();
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
   VMContext *ctx = new VMContext( *m_currentContext );

   // if there are some parameters...
   if ( pSize > 0 )
   {
      // avoid reallocation afterwards.
      ctx->stack().reserve( pSize );
      // copy flat
      for( int32 i = 0; i < pSize; i++ )
      {
         ctx->stack().append( stack()[ stack().length() - pSize + i ] );
      }
      stack().resize( stack().length() - pSize );
   }
   // rotate the context
   m_contexts.pushBack( ctx );

   return ctx;
}


bool VMachine::callCoroFrame( const Item &callable, int32 pSize )
{
   if ( ! callable.isCallable() )
      return false;

   // rotate the context
   putAtSleep( m_currentContext );

   m_currentContext = coPrepare( pSize );

   // fake the frame as a pure return value; this will force this coroutine to terminate
   // without peeking any code in the module.
   m_currentContext->pc_next() = i_pc_call_external_return;
   callFrame( callable, pSize );

   return true;

}

//=====================================================================================
// messages
//

void VMachine::postMessage( VMMessage *msg )
{
   // ok, we can't do it now; post the message
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

void VMachine::processMessage( VMMessage *msg )
{
   // find the slot
   CoreSlot* slot = getSlot( msg->name(), false );
   if ( slot == 0 || slot->empty() )
   {
      msg->onMsgComplete( false );
      delete msg;
   }

   // create the coroutine, whose first operation will be
   // to call our external return frame.
   VMContext* sleepCtx = coPrepare(0);
   
   // force the sleeping context to call the return frame immediately
   sleepCtx->pc_next() = i_pc_call_external_return;
   sleepCtx->pc() = i_pc_call_external_return;

   // prepare the broadcast in the frame.
   slot->prepareBroadcast( sleepCtx, 0, 0, msg );

   // force immediate context rotation
   putAtSleep( m_currentContext );
   m_currentContext = sleepCtx;

   // process immediate execution.
   //callReturn();
}

void VMachine::performGC( bool bWaitForCollect )
{
   m_bWaitForCollect = bWaitForCollect;
   memPool->idleVM( this, true );
   m_eGCPerformed.wait();
}


void VMachine::prepareFrame( CoreFunc* target, uint32 paramCount )
{
   // eventually check for named parameters
   if ( this->regBind().flags() == 0xF0 )
   {
      const SymbolTable *symtab;

      if( target->symbol()->isFunction() )
         symtab = &target->symbol()->getFuncDef()->symtab();
      else
         symtab = target->symbol()->getExtFuncDef()->parameters();

      this->regBind().flags(0);
      // We know we have (probably) a named parameter.
      uint32 size = this->stack().length();
      uint32 paramBase = size - paramCount;
      ItemArray iv(8);

      uint32 pid = 0;

      // first step; identify future binds and pack parameters.
      while( paramBase+pid < size )
      {
         Item &item = this->stack()[ paramBase+pid ];
         if ( item.isFutureBind() )
         {
            // we must move the parameter into the right position
            iv.append( item );
            for( uint32 pos = paramBase + pid + 1; pos < size; pos ++ )
            {
               this->stack()[ pos - 1 ] = this->stack()[ pos ];
            }
            this->stack()[ size-1 ].setNil();
            size--;
            paramCount--;
         }
         else
            pid++;
      }
      this->stack().resize( size );

      // second step: apply future binds.
      for( uint32 i = 0; i < iv.length(); i ++ )
      {
         Item &item = iv[i];

         // try to find the parameter
         const String *pname = item.asLBind();
         Symbol *param = symtab == 0 ? 0 : symtab->findByName( *pname );
         if ( param == 0 ) {
            throw new CodeError( ErrorParam( e_undef_param, __LINE__ ).extra(*pname) );
         }

         // place it in the stack; if the stack is not big enough, resize it.
         if ( this->stack().length() <= param->itemId() + paramBase )
         {
            paramCount = param->itemId()+1;
            this->stack().resize( paramCount + paramBase );
         }

         this->stack()[ param->itemId() + paramBase ] = item.asFBind()->origin();
      }
   }

   // ensure against optional parameters.
   if( target->symbol()->isFunction() )
   {
      FuncDef *tg_def = target->symbol()->getFuncDef();

      if( paramCount < tg_def->params() )
      {
         this->stack().resize( this->stack().length() + tg_def->params() - paramCount );
         paramCount = tg_def->params();
      }

      this->createFrame( paramCount );

      // space for locals
      if ( tg_def->locals() > 0 )
      {
         this->currentFrame()->resizeStack( tg_def->locals() );

         // are part of this locals closed?
         if( target->closure() != 0 ) {
            fassert( target->closure()->length() <= tg_def->locals() );
            this->stack().copyOnto( 0, *target->closure() );
         }
      }

      this->m_currentContext->lmodule( target->liveModule() );
      this->m_currentContext->symbol( target->symbol() );

      //jump
      this->m_currentContext->pc_next() = 0;
   }
   else
   {
      this->createFrame( paramCount );

      // so we can have adequate tracebacks.
      this->m_currentContext->lmodule( target->liveModule() );
      this->m_currentContext->symbol( target->symbol() );

      this->m_currentContext->pc_next() = VMachine::i_pc_call_external;
   }
}

void VMachine::prepareFrame( CoreArray* arr, uint32 paramCount )
{
   fassert( arr->length() > 0 && arr->at(0).isCallable() );
   Item& carr = (*arr)[0];

   uint32 arraySize = arr->length();
   uint32 sizeNow = this->stack().length();
   CoreDict* bindings = arr->bindings();
   bool hasFuture = false;

   // move parameters beyond array parameters
   arraySize -- ; // first element is the callable item.
   if ( arraySize > 0 )
   {
      // first array element is the called item.
      this->stack().resize( sizeNow + arraySize );

      sizeNow -= paramCount;
      for ( uint32 j = sizeNow + paramCount; j > sizeNow; j -- )
      {
         this->stack()[ j-1 + arraySize ] = this->stack()[ j-1 ];
      }

      // push array paramers
      for ( uint32 i = 0; i < arraySize; i ++ )
      {
         Item &itm = (*arr)[i + 1];
         if( itm.isLBind() )
         {
            if ( itm.asFBind() == 0 )
            {
               if ( this->regBind().isNil() && bindings == 0 )
               {
                  // we must create bindings for this array.
                  bindings = arr->makeBindings();
               }

               if ( bindings != 0 )
               {
                  // have we got this binding?
                  Item *bound = bindings->find( *itm.asLBind() );
                  if ( ! bound )
                  {
                     arr->setProperty( *itm.asLBind(), Item() );
                     bound = bindings->find( *itm.asLBind() );
                  }

                  this->stack()[ i + sizeNow ] = *bound;
               }
               else
               {
                  // fall back to currently provided bindings
                  this->stack()[ i + sizeNow ] = *this->getSafeBinding( *itm.asLBind() );
               }
            }
            else {
               // treat as a future binding
               hasFuture = true;
               this->stack()[ i + sizeNow ] = itm;
            }
         }
         else {
            // just transfer the parameters
            this->stack()[ i + sizeNow ] = itm;
         }
      }
   }

   // inform the called about future state
   if( hasFuture )
      this->regBind().flagsOn( 0xF0 );

   carr.readyFrame( this, arraySize + paramCount );

   // change the bindings now, before the VM runs this frame.
   if ( this->regBind().isNil() && arr->bindings() != 0 )
   {
      this->regBind() = arr->bindings();
   }
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

void VMachine::onIdleTime( numeric seconds )
{
   if ( seconds < 0.0 )
   {
      throw new CodeError(
         ErrorParam( e_deadlock ).origin( e_orig_vm ).
            symbol( currentSymbol()->name() ).
            module( currentModule()->name() )
         );
   }

   idle();
   bool complete = m_systemData.sleep( seconds );
   unidle();

   if ( ! complete )
   {
      throw new InterruptedError(
         ErrorParam( e_interrupted ).origin( e_orig_vm ).
            symbol( currentSymbol()->name() ).
            module( currentModule()->name() )
         );
   }
}


void VMachine::handleRaisedItem( Item& value )
{
   // can someone get it?
   if( currentContext()->tryFrame() == 0 )  // uncaught error raised from scripts...
   {
      // create the error that the external application will see.
      Error *err;
      if ( value.isObject() && value.isOfClass( "Error" ) )
      {
         // in case of an error of class Error, we have already a good error inside of it.
         err = static_cast<core::ErrorObject *>(value.asObjectSafe())->getError();
         err->incref();
      }
      else {
         // else incapsulate the item in an error.
         err = new GenericError( ErrorParam( e_uncaught ).origin( e_orig_vm ) );
         err->raised( value );
      }
      err->hasTraceback();
      throw err;
   }

   // Enter the stack frame that should handle the error (or raise to the top if uncaught)
   while( currentFrame()->m_try_base == i_noTryFrame )
   {
      // neutralize post-processors
      // currentFrame()->m_endFrameFunc = 0;  -- done by currentContext()->callReturn();
      m_break = currentContext()->callReturn();
      // let the VM deal with returns
      if ( m_break )
      {
         m_break = false;
         throw value;
      }
   }

   regB() = value;
   // We are in the frame that should handle the error, in one way or another
   // should we catch it?
   popTry( true );
}


void VMachine::handleRaisedError( Error* err )
{
   if( ! err->hasTraceback() )
      fillErrorContext( err, true );

   // catch it if possible
   if( err->catchable() && currentContext()->tryFrame() != 0 )
   {
      // Enter the stack frame that should handle the error (or raise to the top if uncaught)
      while( currentFrame() != currentContext()->tryFrame() )
      {
         // neutralize post-processors
         // currentFrame()->m_endFrameFunc = 0;  -- done by currentFrame()->callReturn();
         m_break = currentContext()->callReturn();
         // let the VM deal with returns
         if ( m_break )
         {
            m_break = false;
            throw err;
         }
      }

      CoreObject *obj = err->scriptize( this );
      if ( obj != 0 )
      {
         err->decref();
         regB() = obj;
         // We are in the frame that should handle the error, in one way or another
         // should we catch it?
         popTry( true );
      }
      else {
         // Panic. Should not happen -- scriptize has raised a symbol not found error
         // describing the missing error class; we must tell the user so that the module
         // not declaring the correct error class, or failing to export it, can be
         // fixed.
         fassert( false );
         throw err;
      }
   }
   // we couldn't catch the error (this also means we're at stackBase() zero)
   // we should handle it then exit
   else {
      // we should manage the error; if we're here, stackBase() is zero,
      // so we are the last in charge
      throw err;
   }
}


void VMachine::periodicChecks()
{
   // pulse VM idle
   if( m_bGcEnabled )
      m_baton.checkBlock();

   if ( m_opLimit > 0 )
   {
      // Bail out???
      if ( m_opCount > m_opLimit )
         return;
      else
         if ( m_opNextCheck > m_opLimit )
            m_opNextCheck = m_opLimit;
   }

   if( ! m_currentContext->atomicMode() )
   {
      if( m_allowYield && ! m_sleepingContexts.empty() && m_opCount > m_opNextContext ) {
         rotateContext();
         m_opNextContext = m_opCount + m_loopsContext;
         if( m_opNextContext < m_opNextCheck )
            m_opNextCheck = m_opNextContext;
      }

      // Periodic Callback
      if( m_loopsCallback > 0 && m_opCount > m_opNextCallback )
      {
         periodicCallback();
         m_opNextCallback = m_opCount + m_loopsCallback;
         if( m_opNextCallback < m_opNextCheck )
            m_opNextCheck = m_opNextCallback;
      }

      // in case of single step:
      if( m_bSingleStep )
      {
         // stop also next op
         m_opNextCheck = m_opCount + 1;
         return; // maintain the event we have, but exit now.
      }

      // perform messages
      m_mtx_mesasges.lock();
      while( m_msg_head != 0 )
      {
         VMMessage* msg = m_msg_head;
         m_msg_head = msg->next();
         if( m_msg_head == 0 )
            m_msg_tail = 0;

         // it is ok if m_msg_tail is left dangling.
         m_mtx_mesasges.unlock();
         if ( msg->error() )
         {
            Error* err = msg->error();
            err->incref();
            delete msg;
            throw err;
         }
         else
            processMessage( msg );  // do not delete msg

         m_mtx_mesasges.lock();
      }
      m_mtx_mesasges.unlock();
   }
}


void VMachine::raiseHardError( int code, const String &expl, int32 line )
{
   Error *err = new CodeError(
         ErrorParam( code, line ).origin( e_orig_vm )
         .hard()
         .extra( expl )
      );

   // of course, if we're not executing, there nothing to raise
   if( currentSymbol() != 0 )
   {
      fillErrorContext( err );
   }

   throw err;
}


void VMachine::launch( const String &startSym, uint32 paramCount )
{
   Item* lItem = 0;

   if( m_mainModule != 0 ) {
      lItem = m_mainModule->findModuleItem( startSym );
   }

   if ( lItem == 0 )
   {
      lItem = findGlobalItem( startSym );
      if( lItem == 0 ) {
         throw new CodeError(
            ErrorParam( e_undef_sym, __LINE__ ).origin( e_orig_vm ).extra( startSym ).
            symbol( "launch" ).
            module( "core.vm" ) );
      }
   }

   /** \todo allow to call classes at startup. Something like "all-classes" a-la-java */
   if ( ! lItem->isCallable() ) {
      throw new CodeError(
            ErrorParam( e_non_callable, __LINE__ ).origin( e_orig_vm ).
               extra( startSym ).
               symbol( "launch" ).
               module( "core.vm" ) );
   }

   // be sure to pass a clean env.
   try
   {
      reset();
      callItem( *lItem, paramCount );
   }
   catch( VMEventQuit&  )
   {
   }
}


void VMachine::bindItem( const String& name, const Item &tgt )
{
   if ( ! regBind().isDict() )
   {
      regBind() = new CoreDict(new LinearDict() );
   }

   CoreDict* cd = regBind().asDict();
   cd->put( Item( new CoreString( name ) ), tgt );
}

void VMachine::unbindItem( const String& name, Item &tgt ) const
{
   fassert( name.size() > 0 );

   if ( name[0] == '.' )
   {
      tgt.setLBind( new CoreString( name, 1 ) );
   }
   else {
      if ( regBind().isDict() )
      {
         if( regBind().asDict()->find( Item( const_cast<String*>(&name) ), tgt ) )
            return;
      }

      // create the lbind.
      tgt.setLBind( new CoreString( name ) );
   }
}


void VMachine::expandTRAV( uint32 count, Iterator& iter )
{
   // we work a bit differently for dictionaries and normal sequences.
   if ( iter.sequence()->isDictionary() )
   {
      if( count == 1 )
      {
         // if we have one variable, we must create a pair holding key and value
         CoreArray* ca = new CoreArray(2);
         ca->items()[0] = iter.getCurrentKey();
         ca->items()[1] = iter.getCurrent();
         ca->length(2);
         *getNextTravVar() = ca;
      }
      else if ( count != 2 )
      {
         throw
            new AccessError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TR*" ) );
      }
      else {
         // we have two vars to decode
         Item* k = getNextTravVar();
         Item* v = getNextTravVar();
         *k = iter.getCurrentKey();
         *v = iter.getCurrent();
      }
   }
   else {
      // for all the other cases, when we have 1 variable we must just set it inside...
      if( count == 1 )
      {
         *getNextTravVar() = iter.getCurrent();
      }
      else {
         // otherwise, we must match the number of variables with the count of sub-variables.
         const Item& current = iter.getCurrent();
         if( ! current.isArray() || current.asArray()->length() != count )
         {
            throw
               new AccessError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TR*" ) );
         }

         CoreArray* source = current.asArray();
         for( uint32 p = 0; p < count; p++ )
         {
            *getNextTravVar() = source->items()[p];
         }
      }
   }
}

Item* VMachine::getNextTravVar()
{
   fassert( m_currentContext->pc_next() % 4 == 0 );

   byte* code = m_currentContext->code();
   fassert( code[ m_currentContext->pc_next() ] == P_NOP );

   int32 type = int( code[ m_currentContext->pc_next()+1 ] );
   m_currentContext->pc_next()+=sizeof( int32 );
   int32 id = *reinterpret_cast< int32 * >( code + m_currentContext->pc_next() );
   m_currentContext->pc_next()+=sizeof( int32 );

   switch( type )
   {
   case P_PARAM_GLOBID:
      return &moduleItem( id );

   case P_PARAM_LOCID:
      return local( id );

   case P_PARAM_PARID:
      return param( id );

   }

   // shall never be here.
   fassert( false );
   return 0;
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

}

/* end of vm.cpp */
