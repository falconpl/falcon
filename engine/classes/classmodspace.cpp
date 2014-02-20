/*
   FALCON - The Falcon Programming Language.
   FILE: classmodspace.cpp

   Handler for dynamically created module spaces.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 05 Feb 2013 18:07:35 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classmodspace.cpp"

#include <falcon/fassert.h>
#include <falcon/vmcontext.h>
#include <falcon/stderrors.h>
#include <falcon/function.h>
#include <falcon/engine.h>
#include <falcon/modspace.h>
#include <falcon/module.h>
#include <falcon/uri.h>
#include <falcon/stdsteps.h>
#include <falcon/stdhandlers.h>
#include <falcon/modloader.h>

#include <falcon/classes/classmodspace.h>


namespace Falcon {

/*#
@class ModSpace
 @optparam path The search path used by the loader functions.
 @optparam srcenc The source encoding used to read textual source files.
 @optparam savePC An entry of @a ModSpace.savePC
 @optparam useSources An entry of @a ModSpace.useSources
@brief Interface for loading modules in a sandbox.

Falcon organizes modules in groups called @i{module spaces}.
A module space is where a main module loads the required dependencies,
and where the loaded modules cooperate in exporting and importing common
symbols.

Each Falcon @a VMProcess has a main ModSpace which is where the modules
are originally loaded. Functions loading modules dynamically, as include(),
create a new module space where the required module can then store its dependencies
and it's exported symbols, so that the new module doesn't pollute the process space.

Once all the modules living in a ModSpace are dropped, and all the references to the
data they exported are removed from the process, the ModSpace unloads the modules and deletes
their data, eventually destroying itself.

Module spaces are organized in a hierarchy that has the process main space as the top one. Each
module space created there after has a parent, which is the module space where a the module
executing the load request resides, or the top space if the code being execute is module-less (i.e.
dynamic code created on the fly). All the modules and globals visible in a parent module space
are also visible to the modules residing in the children spaces, while the children globals are not
visible to the parent or siblings (unless explicitly and dynamically accessed).

In this way, it is possible to load multiple times the same module, which has the same dependencies
and exports the same variables, without generating a symbol export clash, and keeping their state
isolated so that each loaded copy can work separately and autonomously.

@section modspace_loading

Other than organizing the living space of modules, the ModSpace class is also responsible for
dynamic module loading. It provides asynchronous load facility and support; compiling source files
or loading pre-compiled .fam modules can be done through its compiler interface.

*/

namespace {

static int checkEnumParam( const Item& value, int max )
{
   if( ! value.isOrdinal() )
   {
      throw FALCON_SIGN_XERROR( ParamError, e_inv_prop_value, .extra("N") );
   }

   int64 v = value.forceInteger();

   if( v < 0 || v > max )
   {
      throw FALCON_SIGN_XERROR( ParamError, e_inv_prop_value, .extra(String("0<=N<=").N(max)) );
   }

   return static_cast<int>(v);
}


/*#
 @prop savePC ModSpace
 @brief Indicates how the module space should store modules once it compiles them on the fly.

 When the module space loads a module by compiling a source Falcon script, it might automatically
 save the serialized compiled result to a .fam script, so that the next time the same script is
 serached for, the .fam serialized module is de-serialized at a fraction of the time needed to
 compile the original script.

 However, it might be impossible to save the .fam module on a certain location; the default action
 is to ignore the problem. This property can be used to change it.

 It might assume one of the following values:
 - ModSpace.savePC_NEVER: Don't try to save the pre-compiled modules.
 - ModSpace.savePC_TRY: Try to save the the pre-compiled modules, but ignore I/O errors.
 - ModSpace.savePC_MANDATORY: Try to save the pre-compiled modules and throw an error on failure.
 */

static void get_savePC( const Class*, const String&, void* instance, Item& value )
{
   ModSpace* ms = static_cast<ModSpace*>(instance);
   value.setInteger( static_cast<int64>(ms->modLoader()->savePC()) );
}

static void set_savePC( const Class*, const String&, void* instance, const Item& value )
{
   ModSpace* ms = static_cast<ModSpace*>(instance);
   int v = checkEnumParam( value, static_cast<int>(ModLoader::e_save_mandatory) );
   ms->modLoader()->savePC( static_cast<ModLoader::t_save_pc>(v) );
}


/*#
 @prop parent ModSpace
 @brief The parent module space.

 If this module space hasn't any parent, this will be nil.

 @note Modules loaded by a modulespace without any parent can
 access pre-defined symbols and native engine types, but they cannot
 access symbols defined in the core module (as printl), as the core
 module is stored in the process-wide topmost module space.
*/

static void get_parent( const Class* cls, const String&, void* instance, Item& value )
{
   ModSpace* ms = static_cast<ModSpace*>(instance);
   ModSpace* parent = ms->parent();
   if( parent == 0 )
   {
      value.setNil();
   }
   else {
      // for sure, it's already in GC.
      value.setUser(cls,parent);
   }
}



/*#
 @prop checkFTD ModSpace
 @brief Determines how Falcon Template Documents are checked and interpreted.

 When loading a source Falcon script, by default, the module loader decides to interpret
 it as a FTD (Falcon Template Document) if the file has a .ftd extension.

 This setting changes this behavior, and can assume one of the following values:
 - ModSpace.checkFTD_NEVER: All the source files are interpreted as falcon scripts.
 - ModSpace.checkFTD_CHECK: Source files having the .ftd extension will be compiled as FTD scripts.
 - ModSpace.checkFTD_ALWAYS: All the source files are interpreted as FTD scripts.
 */

static void get_checkFTD( const Class*, const String&, void* instance, Item& value )
{
   ModSpace* ms = static_cast<ModSpace*>(instance);
   value.setInteger( static_cast<int64>(ms->modLoader()->checkFTD()) );
}

static void set_checkFTD( const Class*, const String&, void* instance, const Item& value )
{
   ModSpace* ms = static_cast<ModSpace*>(instance);
   int v = checkEnumParam( value, static_cast<int>(ModLoader::e_ftd_force) );
   ms->modLoader()->checkFTD( static_cast<ModLoader::t_check_ftd>(v) );
}


/*#
 @prop useSources ModSpace
 @brief Determines if and when precompiled modules are preferred to source modules.

 When asked to load a source script, the module space can search for a pre-compiled
 version of the same module and load it instead. This property controls how this decision
 is taken, and can assume one of the following values:

 - ModSpace.useSources_NEWER: Use the precompiled .fam module if it's is newer than the source file (the default).
 - ModSpace.useSource_ALWAYS: Always use the source files, ignoring precompiled modules.
 - ModSpace.checkFTD_NEVER: Never use the source files, try to load .fam modules only.

 The result of the checkFTD_NEVER setting is also that all modules that are requested for loading
 will be interpreted as pre-compiled modules, regardless of their extension, and thus, an error will be
 raised if they fail to be deserialized as pre-compiled modules.
*/

static void get_useSources( const Class*, const String&, void* instance, Item& value )
{
   ModSpace* ms = static_cast<ModSpace*>(instance);
   value.setInteger( static_cast<int64>(ms->modLoader()->useSources()) );
}

static void set_useSources( const Class*, const String&, void* instance, const Item& value )
{
   ModSpace* ms = static_cast<ModSpace*>(instance);
   int v = checkEnumParam( value, static_cast<int>(ModLoader::e_us_never) );
   ms->modLoader()->useSources( static_cast<ModLoader::t_use_sources>(v) );
}

/*#
 @prop saveRemote ModSpace
 @brief Tells the ModSpace to try to save precompiled modules also on remote VFS systems.

 Can be either @b true or @b false.
*/

static void get_saveRemote( const Class*, const String&, void* instance, Item& value )
{
   ModSpace* ms = static_cast<ModSpace*>(instance);
   value.setBoolean( ms->modLoader()->saveRemote() );
}

static void set_saveRemote( const Class*, const String&, void* instance, const Item& value )
{
   ModSpace* ms = static_cast<ModSpace*>(instance);
   ms->modLoader()->saveRemote( value.isTrue() );
}



/*#
 @prop senc ModSpace
 @brief Source text encoding used to read the source modules.

 It's a string representing the name of the text encoding used to
 read the source files. It defaults to the source encoding used to
 load the main module of the current process.
*/

static void get_senc( const Class*, const String&, void* instance, Item& value )
{
   ModSpace* ms = static_cast<ModSpace*>(instance);
   String* res = new String( ms->modLoader()->sourceEncoding());
   value = FALCON_GC_HANDLE(res);
}

static void set_senc( const Class*, const String&, void* instance, const Item& value )
{
   ModSpace* ms = static_cast<ModSpace*>(instance);
   if( ! value.isString() )
   {
      throw FALCON_SIGN_XERROR(TypeError, e_inv_prop_value, .extra("S") );
   }

   const String& encoding = *value.asString();
   bool ok = ms->modLoader()->sourceEncoding( encoding );
   if( ! ok )
   {
      throw FALCON_SIGN_XERROR(ParamError, e_inv_prop_value, .extra("Unkonwn encoding " + encoding) );
   }
}


/*#
 @prop famExt ModSpace
 @brief File extension used to search and/or create pre-compiled moldules.

 It's a string representing the file extension (last part of the filename,
 after a final dot) used when reading or writing a pre-compiled module.
*/

static void get_famExt( const Class*, const String&, void* instance, Item& value )
{
   ModSpace* ms = static_cast<ModSpace*>(instance);
   String* res = new String( ms->modLoader()->famExt());
   value = FALCON_GC_HANDLE(res);
}

static void set_famExt( const Class*, const String&, void* instance, const Item& value )
{
   ModSpace* ms = static_cast<ModSpace*>(instance);
   if( ! value.isString() )
   {
      throw FALCON_SIGN_XERROR(TypeError, e_inv_prop_value, .extra("S") );
   }

   const String& v = *value.asString();
   ms->modLoader()->famExt( v );
}


/*#
 @prop ftdExt ModSpace
 @brief File extension used to search for Falcon Template Document source files.
*/

static void get_ftdExt( const Class*, const String&, void* instance, Item& value )
{
   ModSpace* ms = static_cast<ModSpace*>(instance);
   String* res = new String( ms->modLoader()->ftdExt());
   value = FALCON_GC_HANDLE(res);
}

static void set_ftdExt( const Class*, const String&, void* instance, const Item& value )
{
   ModSpace* ms = static_cast<ModSpace*>(instance);
   if( ! value.isString() )
   {
      throw FALCON_SIGN_XERROR(TypeError, e_inv_prop_value, .extra("S") );
   }

   const String& v = *value.asString();
   ms->modLoader()->ftdExt( v );
}



/*#
 @prop path ModSpace
 @brief Search path for required modules.

 The path is a semi-comma separated list of URIs.
*/

static void get_path( const Class*, const String&, void* instance, Item& value )
{
   ModSpace* ms = static_cast<ModSpace*>(instance);
   String* res = new String( ms->modLoader()->getSearchPath());
   value = FALCON_GC_HANDLE(res);
}

static void set_path( const Class*, const String&, void* instance, const Item& value )
{
   ModSpace* ms = static_cast<ModSpace*>(instance);
   if( ! value.isString() )
   {
      throw FALCON_SIGN_XERROR(TypeError, e_inv_prop_value, .extra("S") );
   }

   const String& v = *value.asString();
   ms->modLoader()->setSearchPath( v );
}


static void internal_find_by( VMContext* ctx, bool byName, const String& name )
{
   static Class* clsModule = static_cast<Class*>(Engine::instance()->getMantra("Module"));
   fassert( clsModule != 0 );

   ModSpace* self = static_cast<ModSpace*>(ctx->self().asInst());
   Module* mod = byName? self->findByName( name ) : self->findByURI( name );
   if( mod != 0 )
   {
      mod->incref();
      ctx->returnFrame( FALCON_GC_STORE( clsModule, mod ) );
   }
   else {
      ctx->returnFrame();
   }
}

/*#
 @method findByName ModSpace
 @brief Searches a module stored in the modspace using its logical name.
 @param name The logical module name under which the required module is stored.
 @return A Module instance or nil if not found.

 This method finds a module stored in the ModSpace using the logical name
 under which is stored as a key.

 */

FALCON_DECLARE_FUNCTION(findByName, "name:S");
FALCON_DEFINE_FUNCTION_P1( findByName )
{
   Item* i_name = ctx->param(0);
   if( i_name == 0 || ! i_name->isString() )
   {
      throw paramError( __LINE__, SRC );
   }

   const String& name = *i_name->asString();
   internal_find_by(ctx, true, name);
}

/*#
 @method findByUri ModSpace
 @brief Searches a module stored in the modspace using its URI.
 @param uri The complete URI of the module.
 @return A Module instance or nil if not found.

 This method uses the physical URI (system or network path) that was
 used to load the module as a key to find the module in the ModSpace.

 The URI under which a module is stored is the one used to load it: it might
 be a relative system path in case it was loaded that way.
 */
FALCON_DECLARE_FUNCTION(findByURI, "uri:S|URI");
FALCON_DEFINE_FUNCTION_P1( findByURI )
{
   static Class* clsUri = static_cast<Class*>(Engine::instance()->getMantra("URI"));
   fassert( clsUri != 0 );

   Item* i_name = ctx->param(0);
   void* data = 0;
   Class* cls = 0;
   if( i_name == 0 )
   {
      throw paramError( __LINE__, SRC );
   }
   else if( i_name->isString() )
   {
      const String& name = *i_name->asString();
      internal_find_by(ctx, false, name);
   }
   else if( i_name->asClassInst(cls,data) && cls->isDerivedFrom(clsUri) )
   {
      URI* uri = static_cast<URI*>(data);
      String name = uri->encode();
      internal_find_by(ctx, false, name);
   }
}

static void internal_append_prepend( Function* func, VMContext* ctx, bool append )
{
   static Class* clsUri = static_cast<Class*>(Engine::instance()->getMantra("URI"));
   fassert( clsUri != 0 );
   ModSpace* self = static_cast<ModSpace*>(ctx->self().asInst());

   Item* i_name = ctx->param(0);
   void* data = 0;
   Class* cls = 0;
   if( i_name == 0 )
   {
      throw func->paramError( __LINE__, SRC );
   }
   else if( i_name->isString() )
   {
      const String& name = *i_name->asString();
      if( append ) {
         self->modLoader()->addDirectoryBack( name );
      }
      else {
         self->modLoader()->addDirectoryFront( name );
      }
   }
   else if( i_name->asClassInst(cls,data) && cls->isDerivedFrom(clsUri) )
   {
      URI* uri = static_cast<URI*>(data);
      String name = uri->encode();
      if( append ) {
         self->modLoader()->addDirectoryBack( name );
      }
      else {
         self->modLoader()->addDirectoryFront( name );
      }
   }
}

/*#
 @method appendPath ModSpace
 @brief Appends a path specification to the search path.
 @param uri An URI or a string indicating a single VFS path entry.
 */
FALCON_DECLARE_FUNCTION(appendPath, "uri:S|URI");
FALCON_DEFINE_FUNCTION_P1( appendPath )
{
   internal_append_prepend( this, ctx, true );
}


/*#
 @method prependPath ModSpace
 @brief Prepends a path specification to the search path.
 @param uri An URI or a string indicating a single VFS path entry.
 */
FALCON_DECLARE_FUNCTION( prependPath, "uri:S|URI");
FALCON_DEFINE_FUNCTION_P1( prependPath )
{
   internal_append_prepend( this, ctx, false );
}


static Module* getParentModule( VMContext* ctx )
{
   // else, search it in our context
   Module* ms = 0;

   // try to get the module space of the calling context.
   if( ctx->callDepth() > 1 )
   {
      const CallFrame& frame = ctx->callerFrame(1);
      if( frame.m_function != 0 )
      {
         ms = frame.m_function->module();
      }
   }

   return ms;
}



static ModSpace* getParentModSpace( VMContext* ctx )
{
   // else, search it in our context
   ModSpace* ms = 0;

   // try to get the module space of the calling context.
   if( ctx->callDepth() > 1 )
   {
      const CallFrame& frame = ctx->callerFrame(1);
      if( frame.m_function != 0 && frame.m_function->module() != 0)
      {
         ms = frame.m_function->module()->modSpace();
      }
   }

   if( ms == 0 )
   {
      ms = ctx->process()->modSpace();
   }
   return ms;
}


static void load_internal( Function* caller, VMContext* ctx, bool isUri )
{
   static PStep* step = &Engine::instance()->stdSteps()->m_returnFrameWithTop;
   static Class* clsUri = Engine::instance()->stdHandlers()->uriClass();

   Item* i_name = ctx->param(0);
   Item* i_runMain = ctx->param(2);
   Item* i_asLoad = ctx->param(1);

   bool asLoad = i_asLoad != 0 ? i_asLoad->isTrue() : false;
   bool runMain = i_runMain != 0 ? i_runMain->isTrue() : true;

   Class* cls = 0;
   void* data = 0;
   ModSpace* self = static_cast<ModSpace*>(ctx->self().asInst());
   Module* pmod = getParentModule(ctx);

   if( i_name == 0 )
   {
      throw caller->paramError( __LINE__, SRC );
   }
   else if( i_name->isString() )
   {
      ctx->pushCode( step );
      self->loadModuleInContext(*i_name->asString(), isUri, asLoad, !runMain, ctx, pmod, true );
   }
   else if( i_name->asClassInst(cls,data) && cls->isDerivedFrom(clsUri) )
   {
      ctx->pushCode( step );
      URI* uri = static_cast<URI*>(data);
      self->loadModuleInContext(uri->encode(), isUri, asLoad, !runMain, ctx, pmod, true );
   }
   else
   {
      throw caller->paramError( __LINE__, SRC );
   }

   // don't return the frame, the return step will do.
}

/*#
 @method loadByURI ModSpace
 @brief Loads a module given its phisical name (as an URI).
 @param uri An URI or a string indicating a single VFS path entry.
 @optparam runMain if set to false, the main context will not be run.
 @optparam asLoad if true, exports requests are fulfilled.
 */
FALCON_DECLARE_FUNCTION( loadByURI, "uri:S|URI,runMain:[B],asLoad:[B]");
FALCON_DEFINE_FUNCTION_P1( loadByURI )
{
   load_internal( this, ctx, true );
}

/*#
 @method loadByName ModSpace
 @brief Loads a module given its logical name.
 @param uri An URI or a string indicating a single VFS path entry.
 @optparam runMain if set to false, the main context will not be run.
 @optparam asLoad if true, exports requests are fulfilled.

 The logical name is relative to the module from which loadByName
 is called.

 For example, if the module calling this method is "my.mod",
 loadByName("self.child") will result in searching the module named
 "my.mod.child".
 */
FALCON_DECLARE_FUNCTION( loadByName, "uri:S|URI,runMain:[B],asLoad:[B]");
FALCON_DEFINE_FUNCTION_P1( loadByName )
{
   load_internal( this, ctx, false );
}


static ModSpace* configure_ms(Function* caller, VMContext* ctx, ModSpace* model )
{
   Item* i_path = ctx->param(0);
   Item* i_srcenc = ctx->param(1);
   Item* i_saveFAM = ctx->param(3);
   Item* i_prefer = ctx->param(4);

   if(
        (i_saveFAM != 0 && ! (i_saveFAM->isNil() || i_saveFAM->isOrdinal()) )
        || (i_prefer != 0 && ! (i_prefer->isNil() || i_prefer->isOrdinal()) )
        || (i_srcenc != 0 && ! (i_srcenc->isNil() || i_srcenc->isString()) )
        || (i_path != 0 && ! (i_path->isNil() || i_path->isString()) )
     )
   {
      throw caller->paramError(__LINE__, SRC);
   }

   ModSpace* ms = new ModSpace(ctx->process());

   if( i_saveFAM != 0 && ! i_saveFAM->isNil() )
   {
      ms->modLoader()->savePC( (ModLoader::t_save_pc) checkEnumParam(*i_saveFAM, (int) ModLoader::e_save_mandatory ) );
   }
   else if( model != 0 ){
      ms->modLoader()->savePC( model->modLoader()->savePC() );
   }


   if( i_prefer != 0 && ! i_prefer->isNil() )
   {
      ms->modLoader()->useSources( (ModLoader::t_use_sources) checkEnumParam(*i_prefer, (int) ModLoader::e_save_mandatory ) );
   }
   else if( model != 0 ){
      ms->modLoader()->useSources( model->modLoader()->useSources() );
   }

   if( i_srcenc != 0 && ! i_srcenc->isNil() )
   {
      const String& srcenc = *i_srcenc->asString();
      if( Engine::instance()->getTranscoder(srcenc) == 0 )
      {
         throw FALCON_SIGN_XERROR(ParamError, e_param_range,
               .extra(String("Unknown encoding ").A(srcenc) ) );
      }
      ms->modLoader()->sourceEncoding(srcenc);
   }
   else if( model != 0 ){
      ms->modLoader()->sourceEncoding( model->modLoader()->sourceEncoding() );
   }


   if( i_path != 0 && ! i_path->isNil() )
   {
      const String& path = *i_path->asString();
      ms->modLoader()->setSearchPath(path);
   }
   else if( model != 0 ){
      ms->modLoader()->setSearchPath( model->modLoader()->getSearchPath() );
   }

   if( model != 0 )
   {
      ms->modLoader()->checkFTD(model->modLoader()->checkFTD());
      ms->modLoader()->ftdExt(model->modLoader()->ftdExt());
      ms->modLoader()->famExt(model->modLoader()->famExt());
      ms->modLoader()->saveRemote(model->modLoader()->saveRemote());
   }

   return ms;
}


FALCON_DECLARE_FUNCTION(init, "path:[S],srcenc:[S],savePC:[N],useSources:[N]")
FALCON_DEFINE_FUNCTION_P1(init)
{
   // the model MS is the MS of the calling module
   ModSpace* modelMS = getParentModSpace(ctx);
   ModSpace* ms = configure_ms(this, ctx, modelMS);
   ctx->self() = FALCON_GC_STORE(this->methodOf(),ms);
   ctx->returnFrame(ctx->self());
}

/*#
 @method makeChild ModSpace
 @brief Create a child module space.
 @optparam path The search path used by the loader functions.
 @optparam srcenc The source encoding used to read textual source files.
 @optparam savePC An entry of @a ModSpace.savePC
 @optparam useSources An entry of @a ModSpace.useSources
 @return A new module space, child of this one.

 This method creates a new module space that has this module space as parent.
 The child module space can independently load modules, and symbols exported there
 are visible to the locally loaded module only. However, every module loaded in the
 child space has full access to the modules and global symbols available in the
 parent space.

 Modules in the child space can override exports and unique module IDs defined in the
 parent, making their own version the one visible to all the other modules loaded
 in the same space, but leaving the parent space untainted.

 The child module space inherits all the settings of the parent, unless otherwise specified
 in the parameters.
*/
FALCON_DECLARE_FUNCTION(makeChild, "path:[S],srcenc:[S],savePC:[N],useSources:[N]")
FALCON_DEFINE_FUNCTION_P1(makeChild)
{

   ModSpace* self = ctx->tself<ModSpace*>();
   ModSpace* ms = configure_ms(this, ctx, self);
   ms->setParent( self );
   ctx->returnFrame(FALCON_GC_STORE(this->methodOf(),ms));
}

/*#
 @method getExport ModSpace
 @brief Searches for values globally exported by the module(s) in the module space.
 @param varname The name under which the variable was exported
 @optparam dflt A default value to be returned if the variable is not found.
 @raise AccessError if the variable is not found and @b dflt is not given.

 */
FALCON_DECLARE_FUNCTION(getExport, "varname:S,dflt:[X]")
FALCON_DEFINE_FUNCTION_P1(getExport)
{
   Item* i_varname = ctx->param(0);
   Item* i_dflt = ctx->param(1);
   if (i_varname == 0 || ! i_varname->isString() )
   {
      throw paramError(__LINE__, SRC);
   }

   const String& varname = *i_varname->asString();
   ModSpace* ms = ctx->tself<ModSpace*>();
   Item* val = ms->findExportedValue(varname);

   if( val != 0 )
   {
      ctx->returnFrame(*val);
   }
   else if( i_dflt != 0 )
   {
      ctx->returnFrame(*i_dflt);
   }
   else {
      throw FALCON_SIGN_XERROR( AccessError, e_undef_sym, .extra(varname) );
   }
}

/*#
 @method setExport ModSpace
 @brief Adds or modifies an exported value.
 @param varname The name under which the variable was exported
 @param value The value to be set.
 @raise AccessError if the variable is not found and @b dflt is not given.
 */
FALCON_DECLARE_FUNCTION(setExport, "varname:S,value:X")
FALCON_DEFINE_FUNCTION_P1(setExport)
{
   Item* i_varname = ctx->param(0);
   Item* i_value= ctx->param(1);
   if (i_varname == 0 || ! i_varname->isString() || i_value == 0)
   {
      throw paramError(__LINE__, SRC);
   }

   ModSpace* ms = ctx->tself<ModSpace*>();
   const String& varname = *i_varname->asString();
   ms->setExportValue(varname, *i_value);
   ctx->returnFrame();
}

}


ClassModSpace::ClassModSpace():
         Class("ModSpace")
{
   addProperty( "savePC", &get_savePC, &set_savePC );
   addProperty( "checkFTD", &get_checkFTD, &set_checkFTD );
   addProperty( "useSources", &get_useSources, &set_useSources );
   addProperty( "saveRemote", &get_saveRemote, &set_saveRemote );
   addProperty( "senc", &get_senc, &set_senc );
   addProperty( "famExt", &get_famExt, &set_famExt );
   addProperty( "ftdExt", &get_ftdExt, &set_ftdExt );
   addProperty( "path", &get_path, &set_path );
   // don't show the parent property in standard description.
   addProperty( "parent", &get_parent, 0,false, true );

   addConstant( "savePC_NEVER", static_cast<int64>(ModLoader::e_save_no) );
   addConstant( "savePC_TRY", static_cast<int64>(ModLoader::e_save_try) );
   addConstant( "savePC_MANDATORY", static_cast<int64>(ModLoader::e_save_mandatory) );

   addConstant( "checkFTD_NEVER", static_cast<int64>(ModLoader::e_ftd_ignore) );
   addConstant( "checkFTD_CHECK", static_cast<int64>(ModLoader::e_ftd_check) );
   addConstant( "checkFTD_ALWAYS", static_cast<int64>(ModLoader::e_ftd_force) );

   addConstant( "useSources_NEWER", static_cast<int64>(ModLoader::e_us_newer) );
   addConstant( "useSources_ALWAYS", static_cast<int64>(ModLoader::e_us_always) );
   addConstant( "useSources_NEVER", static_cast<int64>(ModLoader::e_us_never) );

   setConstuctor( new Function_init );

   addMethod( new Function_findByName );
   addMethod( new Function_findByURI );
   addMethod( new Function_appendPath );
   addMethod( new Function_prependPath );
   addMethod( new Function_makeChild );

   addMethod( new Function_getExport );
   addMethod( new Function_setExport );

   addMethod( new Function_loadByName );
   addMethod( new Function_loadByURI );
}

ClassModSpace::~ClassModSpace()
{
}

void* ClassModSpace::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}

void ClassModSpace::dispose( void* instance ) const
{
   ModSpace* spc = static_cast<ModSpace*>(instance);
   spc->decref();
}

void* ClassModSpace::clone( void* instance ) const
{
   ModSpace* spc = static_cast<ModSpace*>(instance);
   spc->incref();
   return spc;
}

void ClassModSpace::gcMarkInstance( void* instance, uint32 mark ) const
{
   ModSpace* spc = static_cast<ModSpace*>(instance);
   spc->gcMark(mark);
}

bool ClassModSpace::gcCheckInstance( void* instance, uint32 mark ) const
{
   ModSpace* spc = static_cast<ModSpace*>(instance);
   return spc->currentMark() >= mark;
}
   
}

/* end of classmodspace.cpp */
