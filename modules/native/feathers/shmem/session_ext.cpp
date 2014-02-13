/*
   FALCON - The Falcon Programming Language.
   FILE: session_ext.cpp

   Falcon script interface for Inter-process persistent data.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 15 Nov 2013 15:31:11 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/modules/native/feathers/shmem/session_ext.cpp"

#include "session_ext.h"
#include "session.h"
#include "errors.h"

#include <falcon/trace.h>
#include <falcon/function.h>
#include <falcon/vmcontext.h>
#include <falcon/stdsteps.h>
#include <falcon/stderrors.h>
#include <falcon/symbol.h>
#include <falcon/itemdict.h>

/*#  @beginmodule shmem */

namespace Falcon {

namespace {

/*#
@class Session
@brief Automatism for using persistent data
@param id The session ID or name
@param mode Whether to use shared memory with or without backup, or a plain file.

Session class provides an automatism to save and restore persistent data,
concurrently from different processes.

Practically, it serializes a set of data on a semi-persistent or persistent
storage, eventually taking care of applying the stored data to locally visible
symbols in the current context. It also supports session timeout and expiration.

The mode parameter can be one of the following:

   - OM_SHMEM: (the default) The session is opened in shared memory.
   - OM_SHMEM_BU: The session is opened as a plain file, backed up in shared memory.
   - OM_FILE: The session data is stored and saved on a regular system file.

When the session is backed-up on a file, the ID parameter is also used as the
filename where the session is stored.

The session object can be queried directly in order to manipulate the data that
is stored in the session.


@section session_usage Usage pattern

The method @a Session.start is the most common way to ready a session. It accepts
a list of symbols that are then queried in the current context; if the session was
already opened, the values are restored and applied directly to the given symbols.

Before the program terminates, the user should invoke @a Session.save to grab the
values currently held in the variables, and store them on the permanent media.

While it is possible to pass symbols that don't refer to an existing variable,
it is avisable to have already declared at least one of the symbols stored in the
session when it's started, so that, if the session didn't exist, the program can
detect this by checking if an arbitrary value was placed on that control variable.

@code

// initialize an arbitrary variable
control = nil

sobj = Session("MySession")
sobj.start( &control, &var1, &var2, &var3 )

if not control
   > "Initializing the session"
   control = true
   var1 = 0
   //... set var2/var3 to desired initial values
else
   > "Var1 value: ", var1
   > "Var2 value: ", var2
   > "Var3 value: ", var3
   // Modify varN values as required
end

// record the changes and save the session.
sobj.save()

@endcode

*/

FALCON_DECLARE_FUNCTION(init, "id:S,mode:[N]")
FALCON_DEFINE_FUNCTION_P1(init)
{
   TRACE1("Session.init() with %d parameters", ctx->paramCount() );

   Item* i_id = ctx->param(0);
   Item* i_mode = ctx->param(1);

   if( i_id == 0 || ! i_id->isString()
       || ( i_mode != 0 && ! i_mode->isOrdinal()) )
   {
      throw paramError(__LINE__, SRC);
   }

   int mode = (int) (i_mode != 0 ? i_mode->forceInteger() : (int) Session::e_om_shmem );
   if( mode != static_cast<int>(Session::e_om_shmem)
       && mode != static_cast<int>(Session::e_om_shmem_bu)
       && mode != static_cast<int>(Session::e_om_file)
   )
   {
      throw paramError( "Invalid mode", __LINE__, SRC);
   }

   Session* session = ctx->tself<Session*>();
   const String& id = *i_id->asString();
   session->setID(id);
   session->setOpenMode(static_cast<Session::t_openmode>(mode));

   TRACE1("Session.init(\"%s\", %d)", id.c_ize(), mode );

   ctx->returnFrame(ctx->self());
}


static void internal_add_remove( Function* func, VMContext* ctx, int32 pCount, bool bAdd, int minCount = 1 )
{
   if( pCount < minCount )
   {
      throw func->paramError(__LINE__);
   }

   Session* session = ctx->tself<Session*>();

   for(int32 i = 0; i < pCount; i++ )
   {
      Item* item = ctx->param(i);
      fassert( item != 0 );
      if( item->isString() )
      {
         const String& symName = *item->asString();
         const Symbol* sym = Engine::getSymbol(symName);
         if( bAdd ) {
            session->addSymbol(sym);
         }
         else {
            session->removeSymbol(sym);
         }
         sym->decref();
      }
      else if( item->isSymbol() )
      {
         const Symbol* sym = item->asSymbol();
         if( bAdd ) {
            session->addSymbol(sym);
         }
         else {
            session->removeSymbol(sym);
         }
      }
      else
      {
         throw func->paramError(String("Parameter ").N(i+1).A(" must be a symbol or a string"));
      }
   }
}

/*#
 @method add Session
 @brief Adds arbitrary symbols to a session.
 @param symbol a symbol (or a symbol name in a string) to be added to the session recording.
 @optparam ... More symbols to be added.

 This method adds one or more arbitrary symbol (either as a string representing a symbol name, or
 a proper symbol object) to the session recording.

*/

FALCON_DECLARE_FUNCTION(add, "symbol:S|Symbol,...")
FALCON_DEFINE_FUNCTION_P(add)
{
   TRACE1("Session.add() with %d parameters", ctx->paramCount() );
   internal_add_remove( this, ctx, pCount, true );
   ctx->returnFrame();
}

/*#
 @method remove Session
 @brief Removes arbitrary symbols to a session.
 @param symbol a symbol (or a symbol name in a string) to be removed from the session recording.
 @optparam ... More symbols to be added.

 This method removes one or more arbitrary symbol (either as a string representing a symbol name, or
 a proper symbol object) from the session recording.

*/
FALCON_DECLARE_FUNCTION(remove, "symbol:S|Symbol,...")
FALCON_DEFINE_FUNCTION_P(remove)
{
   TRACE1("Session.remove() with %d parameters", ctx->paramCount() );
   internal_add_remove( this, ctx, pCount, false );
   ctx->returnFrame();
}

/*#
 @method open Session
 @brief Open an existing session.
 @param apply If not given, or given and true, applies all the values recorded in the session to the current context.
 @raise SessionError If the session is not valid, expired or wasn't previously created with create() or start().

 This method opens an existing session, giving the caller the ability to apply the values in the
 session to the current context upon request.
*/
FALCON_DECLARE_FUNCTION(open, "apply:[B]")
FALCON_DEFINE_FUNCTION_P(open)
{
   Session* session = ctx->tself<Session*>();
   bool bApply = pCount > 0 ? ctx->param(0)->isTrue() : true;
   TRACE1("Session.open(%s)", bApply ? "true" : "false" );

#ifndef NDEBUG
   String id;
   session->getID(id);
   TRACE1("Session(%s).open(%s)", id.c_ize(), (bApply ? "true" : "false") );
#endif

   session->open();
   try {
      session->load(ctx, bApply);
   }
   catch(SessionError* se)
   {
      if (se->errorCode() == FALCON_ERROR_SHMEM_SESSION_NOTOPEN )
      {
         se->decref();
         ctx->returnFrame(Item().setBoolean(false));
      }
      else {
         throw;
      }
   }

   // don't return the frame
}

/*#
 @method create Session
 @brief Creates a new session (eventually destroying previous ones).
 @optparam symbol A symbol object or name to be recored by this session.
 @optparam ... More symbols

 This method creates a new session under the given session ID, possibly
 destroying existing sessions having the same ID. The set of locally visible
 symbols to be recorded must be given as parameters (later on, @a Session.add and
 @a Session.remove can be used to change the symbols saved in the session).

*/
FALCON_DECLARE_FUNCTION(create, "symbol:[S|Symbol],..." )
FALCON_DEFINE_FUNCTION_P(create)
{
   TRACE1("Session.create() with %d parameters", ctx->paramCount() );
   Session* session = ctx->tself<Session*>();

#ifndef NDEBUG
   String id;
   session->getID(id);
   TRACE1("Session(%s).create() with %d parameters", id.c_ize(), ctx->paramCount() );
#endif

   internal_add_remove( this, ctx, pCount, true, 0 );

   session->create();
   ctx->returnFrame();
}

/*#
 @method start Session
 @brief Open an existing session or create a new one if needed.
 @optparam symbol A symbol object or name to be recored by this session.
 @optparam ... More symbols
 @raise SessionError If the session is not valid or expired.

 This method tries to open an existing session with the given ID; if
 a session with the given ID cannot be opened, a new session is created
 on the spot.

 If the session exists, the recorded symbols are immediately applied
 to the current context at current visibility scope (as if a successful
 @a Session.open(true) was invoked).

 The set of locally visible
 symbols to be recorded must be given as parameters (later on, @a Session.add and
 @a Session.remove can be used to change the symbols saved in the session).

*/
FALCON_DECLARE_FUNCTION(start, "symbol:[S|Symbol],...")
FALCON_DEFINE_FUNCTION_P(start)
{
   Session* session = ctx->tself<Session*>();
   TRACE1("Session.start() with %d parameters", ctx->paramCount() );

   try {
      session->open();
      session->load(ctx, true);
      // don't return frame.
   }
   catch( IOError* ie )
   {
      session->close();
      internal_add_remove( this, ctx, pCount, true, 0 );
      session->create();
      ie->decref();
      ctx->returnFrame();
   }
   catch( ShmemError* se )
   {
      session->close();
      internal_add_remove( this, ctx, pCount, true, 0 );
      session->create();
      se->decref();
      ctx->returnFrame();
   }
   catch(SessionError* se)
   {
      if (se->errorCode() == FALCON_ERROR_SHMEM_SESSION_NOTOPEN )
      {
         session->close();
         internal_add_remove( this, ctx, pCount, true, 0 );
         session->create();
         se->decref();
         ctx->returnFrame();
      }
      else {
         throw;
      }
   }
}

/*#
 @method close Session
 @brief Closes a session, destroying it and making it unavailable for further operations.
*/
FALCON_DECLARE_FUNCTION(close, "")
FALCON_DEFINE_FUNCTION_P1(close)
{
   MESSAGE("Session.close()" );
   Session* session = ctx->tself<Session*>();
   session->close();

   ctx->returnFrame();
}


/*#
 @method apply Session
 @brief Applies the symbol values stored in the session to the current context at current visibility scope.
*/
FALCON_DECLARE_FUNCTION(apply, "")
FALCON_DEFINE_FUNCTION_P1(apply)
{
   MESSAGE("Session.apply()" );
   Session* session = ctx->tself<Session*>();
   // we must operate in caller's frame.
   ctx->returnFrame();

   session->apply(ctx);
}

/*#
 @method record Session
 @brief Reads the symbol values from the current context and stores them in the session object.
*/
FALCON_DECLARE_FUNCTION(record, "")
FALCON_DEFINE_FUNCTION_P1(record)
{
   MESSAGE("Session.apply()" );
   Session* session = ctx->tself<Session*>();
   // we must operate in caller's frame.
   ctx->returnFrame();

   session->record(ctx);
}


/*#
 @method save Session
 @brief Store the current contents of the session to the persistent media.
 @optparam record If not given, or if given and true, records the values of the symbols from the current context.

 This method serializes the values stored in the session object, eventually (and tipically)
 fetching them from the current context prior performing the serialization.

 If any of the values to be saved cannot be serialized, an UnserializableError is raised.
*/
FALCON_DECLARE_FUNCTION(save, "record:[B]")
FALCON_DEFINE_FUNCTION_P1(save)
{
   static PStep* retStep = &Engine::instance()->stdSteps()->m_returnFrame;

   Item* i_record = ctx->param(0);
   TRACE("Session.save(%s)", i_record == 0 || i_record->isTrue() ? "true" : "false" );
   Session* session = ctx->tself<Session*>();

   if( i_record == 0 || i_record->isTrue() )
   {
      session->record(ctx);
   }

   ctx->pushCode( retStep );
   session->save(ctx);
   // don't return the frame
}

/*#
 @method get Session
 @brief Retrieves the value of the given symbol in this session.
 @param symbol A symbol (or symbol name) to be searched in this session.
 @optparam dflt A default value to be returned if the symbol is not found.
 @raise AccessError if the given symbol is not found, and @b dflt parameter is not given.

*/
FALCON_DECLARE_FUNCTION(get, "symbol:S|Symbol,dflt:[X]")
FALCON_DEFINE_FUNCTION_P1(get)
{
   MESSAGE("Session.get()" );
   const Symbol* sym;

   Item* i_symbol = ctx->param(0);
   Item* i_dflt = ctx->param(1);

   if( i_symbol == 0 )
   {
      throw paramError(__LINE__, SRC);
   }
   else if( i_symbol->isString() )
   {
      sym = Engine::getSymbol(*i_symbol->asString());
   }
   else if( i_symbol->isSymbol() )
   {
      sym = i_symbol->asSymbol();
   }
   else {
      throw paramError(__LINE__, SRC);
   }

   TRACE("Session.get(\"%s\")", sym->name().c_ize() );
   Item value;
   Session* session = ctx->tself<Session*>();
   bool found = session->get(sym, value);

   if ( i_symbol->isString() )
   {
      sym->decref();
   }

   if ( found )
   {
      ctx->returnFrame( value );
   }
   else {
      if( i_dflt != 0 )
      {
         ctx->returnFrame(*i_dflt);
      }
      else {
         throw FALCON_SIGN_XERROR( AccessError, e_dict_acc, .extra("in session") );
      }
   }
}

/*#
 @method set Session
 @brief Writes a value directly in the session.
 @param symbol A symbol (or symbol name) to be updated in this session.
 @optparam value The value to be written for the given symbol.

 This method updates, or eventually creates a value to be associated
 with the given @b symbol in this session.

*/
FALCON_DECLARE_FUNCTION(set, "symbol:S|Symbol,value:X")
FALCON_DEFINE_FUNCTION_P1(set)
{
   MESSAGE("get.set()" );
   const Symbol* sym;

   Item* i_symbol = ctx->param(0);
   Item* i_value = ctx->param(1);

   if( i_symbol == 0 || i_value == 0 )
   {
      throw paramError(__LINE__, SRC);
   }
   else if( i_symbol->isString() )
   {
      sym = Engine::getSymbol(*i_symbol->asString());
   }
   else if( i_symbol->isSymbol() )
   {
      sym = i_symbol->asSymbol();
   }
   else {
      throw paramError(__LINE__, SRC);
   }

   TRACE("Session.set(\"%s\", ...)", sym->name().c_ize() );
   Item value;
   Session* session = ctx->tself<Session*>();
   session->addSymbol(sym, *i_value);

   ctx->returnFrame();
}

/*#
 @method getAll Session
 @brief Retrieves all the symbol/value pairs stored in this session as a dictionary.
*/

FALCON_DECLARE_FUNCTION(getAll, "")
FALCON_DEFINE_FUNCTION_P1(getAll)
{
   MESSAGE("getAll()");
   Session* session = ctx->tself<Session*>();

   ItemDict* dict = new ItemDict;

   class Rator: public Session::Enumerator
   {
   public:
      Rator( ItemDict* dict ): m_dict(dict) {}
      virtual ~Rator() {}
      virtual void operator()(const Symbol* sym, Item& value)
      {
         m_dict->insert(FALCON_GC_HANDLE(new String(sym->name()) ), value);
      }

   private:
      ItemDict* m_dict;
   };

   Rator rator(dict);

   session->enumerate(rator);
   ctx->returnFrame(FALCON_GC_HANDLE(dict));
}

/*#
 @property timeout Session
 @brief Session timeout in seconds

 If this property is zero, the session never expires. If it's
 nonzero, the session expires after the given amount of seconds.

 A session will expire if it's not read or written for the given amount
 of seconds.

 Changing the value of this property will also reset the expire time,
 so that it will be set to the current moment plus the given timeout.
 */
static void get_timeout( const Class*, const String&, void *instance, Item& value )
{
   Session* session = static_cast<Session*>(instance);
   value.setInteger(session->timeout());
}


static void set_timeout( const Class*, const String&, void *instance, const Item& value )
{
   Session* session = static_cast<Session*>(instance);
   session->timeout(value.forceInteger());
}

/*#
 @property createdAt Session
 @brief Seconds since epoch when this session was created.

 This property is read-only.
 */
static void get_createdAt( const Class*, const String&, void *instance, Item& value )
{
   Session* session = static_cast<Session*>(instance);
   value.setInteger(session->createdAt());
}

/*#
 @property expiresAt Session
 @brief Seconds since epoch when this session is due to expire.

 This property is read-only.

 This value is expressed as seconds since epoch, and is computed by
 adding the value of @a Session.timeout to the last moment when the
 session was accessed or saved.

 If @a Session.timeout is 0, this value will be 0 too.

 */
static void get_expiresAt( const Class*, const String&, void *instance, Item& value )
{
   Session* session = static_cast<Session*>(instance);
   value.setInteger(session->expiresAt());
}

/*#
 @property id Session
 @brief ID of this session (as given in the constructor)

 This property is read-only.
 */
static void get_id( const Class*, const String&, void *instance, Item& value )
{
   Session* session = static_cast<Session*>(instance);
   value = FALCON_GC_HANDLE( &(new String( session->id() ))->bufferize() );
}

}

//=============================================================================
// Session class handler
//=============================================================================

ClassSession::ClassSession():
         Class("Session")
{
   setConstuctor(new FALCON_FUNCTION_NAME(init) );
   addMethod( new FALCON_FUNCTION_NAME(add) );
   addMethod( new FALCON_FUNCTION_NAME(remove) );

   addMethod( new FALCON_FUNCTION_NAME(open) );
   addMethod( new FALCON_FUNCTION_NAME(create) );
   addMethod( new FALCON_FUNCTION_NAME(start) );
   addMethod( new FALCON_FUNCTION_NAME(close) );
   addMethod( new FALCON_FUNCTION_NAME(apply) );
   addMethod( new FALCON_FUNCTION_NAME(record) );
   addMethod( new FALCON_FUNCTION_NAME(save) );

   addMethod( new FALCON_FUNCTION_NAME(get) );
   addMethod( new FALCON_FUNCTION_NAME(set) );
   addMethod( new FALCON_FUNCTION_NAME(getAll) );

   addProperty( "timeout", &get_timeout, &set_timeout );
   addProperty( "createdAt", &get_createdAt );
   addProperty( "expiresAt", &get_expiresAt );
   addProperty( "id", &get_id );

   addConstant("OM_FILE", static_cast<int64>(Session::e_om_file) );
   addConstant("OM_SHMEM", static_cast<int64>(Session::e_om_shmem) );
   addConstant("OM_SHMEM_BU", static_cast<int64>(Session::e_om_shmem_bu) );
}


ClassSession::~ClassSession()
{
}


int64 ClassSession::occupiedMemory( void* inst ) const
{
   Session* s = static_cast<Session*>(inst);

   return s->occupiedMemory();
}


void* ClassSession::createInstance() const
{
   return new Session;
}


void ClassSession::dispose( void* instance ) const
{
   Session* s = static_cast<Session*>(instance);
   delete s;
}


void* ClassSession::clone( void* ) const
{
   return 0;
}

void ClassSession::describe( void* instance, String& target, int depth, int maxlen ) const
{
   //TODO
   Class::describe(instance, target, depth, maxlen);
}

}

/* end of session_ext.cpp */
