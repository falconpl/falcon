/*
   FALCON - The Falcon Programming Language.
   FILE: wopi_ext.h

   Web Oriented Programming Interface

   Object encapsulating requests.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2007-07-28 18:55:17

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/wopi/wopi.h>
#include <falcon/wopi/modulewopi.h>
#include <falcon/itemdict.h>
#include <falcon/itemarray.h>
#include <falcon/vmcontext.h>

#include <falcon/engine.h>
#include <falcon/stdhandlers.h>
#include <stdlib.h>
#include <falcon/wopi/version.h>

/*#
   @beginmodule WOPI
*/

namespace Falcon {
namespace WOPI {

class PStepAfterPersist: public PStep
{
public:
   inline PStepAfterPersist() { apply = apply_; }
   inline virtual ~PStep() {}
   virtual void describeTo( String& target ) const { target = "PStepAfterPersist"; }

   static void apply_(const PStep*, VMContext* ctx)
   {
      // we have the object to be saved in the persist data in top,
      // and the frame is that of the persist() method
      Falcon::Item *i_id = ctx->param( 0 );
      Wopi* wopi = static_cast<Wopi*>(ctx->self().asInst());
      const String& src = *i_id->asString();
      wopi->setContextData( src, ctx->topData() );
      ctx->returnFrame();
   }

};

FALCON_DECLARE_ERROR_INSTANCE( PersistError );
FALCON_DECLARE_ERROR_CLASS_EX( PersistError, \
         addConstant("NotFound", FALCON_ERROR_WOPI_PERSIST_NOT_FOUND );\
         );

/*#
@object WOPI
@brief General configuration and WOPI-system wide interface

This object gives access to generic configuration settings and global
module functions.
*/

//===================================================================================
// Wopi class
//

ClassWOPI::ClassWOPI()
{
   m_stepAfterPersist = new PStepAfterPersist;
}

ClassWOPI::~ClassWOPI()
{
   delete m_stepAfterPersist;
}

void ClassWOPI::dispose( void* ) const
{
   // do nothing
}

virtual void* ClassWOPI::clone( void* ) const
{
   return 0;
}

void* ClassWOPI::createInstance() const
{
   // static class
   return 0;
}


static void internal_htmlEscape_stream( const Falcon::String &str, Falcon::TextWriter *out )
{
   for ( Falcon::uint32 i = 0; i < str.length(); i++ )
   {
      Falcon::uint32 chr = str[i];
      switch ( chr )
      {
         case '<': out->write( "&lt;" ); break;
         case '>': out->write( "&gt;" ); break;
         case '"': out->write( "&quot;" ); break;
         case '&': out->write( "&amp;" ); break;
         default: out->putChar( chr ); break;
      }
   }
}


/*#
   @global scriptName
   @brief Physical name of the loaded main script.

   This is the name of the script that has been loaded and served through the
   web server request. It may be different from the file name that has been
   actually requested, as the web server may perform several mappings.

   scriptPath + "/" + scriptName is granted to give the complete file path
   where the script has been loaded.
*/

static void get_scriptName(const Class* cls, const String&, void* instance, Item& value )
{
   TRACE1( "WOPI.scriptName for %p", instance );
   ModuleWopi* modwopi = static_cast<ModuleWopi*>( cls->module() );
   value = FALCON_GC_HANDLE(new String(modwopi->scriptName()));
}

/*#
   @global scriptPath
   @brief Physical path of the loaded main script.

   This is the directory where the script that has been served through the
   web server request has been loaded from.
   It may be different from the path requested, as the web server may perform
   several path mappings.

   scriptPath + "/" + scriptName is granted to give the complete file path
   where the script has been loaded.

   This information can be useful to the scripts to know the base directory of
   their main code, and use that to perform relative operations on the filesystem.

   @note scriptPath won't be changed through submodules; it stays the same all through
   one execution.
*/

/*#
   @object Wopi
   @brief General framework data and functions
   
   This object allows to access to general purpose functions
   exposed by the framework, and to global framework settings.
   
   A few utility function exposed by the Wopi model are 
   directly exported to the main namespace for convenience
   and because they are widely used. Every other functionality
   is exposed through @a Request, @a Reply or @a Wopi.
*/




/*#
   @method getAppData Wopi
   @brief Gets global, persistent web application data.
   @optparam app Application name for the specific data.
   
   This method restores application-wide data set through
   the @a Wopi.setData method. The optional parameter @b app
   may be specified to indicate a different applicaiton
   name under which the data are to be saved and restored.

   See @a wopi_appdata for further details.
*/


/*#
   @method setAppData Wopi
   @brief Sets global, persistent web application data.
   @param data The data to be saved (in a dictionary)
   @optparam app Application name for the specific data.
   @return True if the synchronization with the application
           data was successful, false in case the data was
           already changed.
   
   This method saves application-wide data, that can be
   retrieved elsewhere to have persistent data. The
   optional parameter @b app
   may be specified to indicate a different applicaiton
   name under which the data are to be saved and restored.
   
   @note Important: be sure that the data to be saved can be
   safely and directly serialized.

   The function may return false if the synchronization failed.
   If you want to atomically receive the new contents of the
   data, pass it by reference.

   See @a wopi_appdata for further details.
*/


/*#
   @method persist Wopi
   @brief Retrieves or eventually creates per-context persistent data
   @param id Unique ID for persistent data.
   @optparam creator Code or callable evaluated to create the persistent data.
   @return The previously saved item.
   @raise PersistError if the persistent key is not found and creator code is not given.

   This method restores execution-context (usually process or thread) specific
   persistent data that was previously saved, possibly by another VM process
   (i.e. another script).

   An optional @b code parameter is evaluated in case the data under
   the given @b id is not found; the evaluation result is
   then stored in the persistent data slot, as if @a Wopi.setPersist was
   called with the same @b id to save the data, and is then returned to
   the caller of this method.

   See @a wopi_pdata for further details.
*/
FALCON_DECLARE_FUNCTION(persist, "id:S,code:[C]")
FALCON_DEFINE_FUNCTION_P1(persist)
{
   // Get the ID of the persistent data
   Falcon::Item *i_id = ctx->param( 0 );
   Falcon::Item *i_func = ctx->param( 1 );

   // parameter sanity check.
   if ( i_id == 0 || ! i_id->isString() ||
         ( i_func != 0 && ! (i_func->isCallable() || i_func->isTreeStep()) )
      )
   {
      throw paramError();
   }

   Wopi* wopi = static_cast<Wopi*>(ctx->self().asInst());

   Item target;
   bool success = wopi->getContextData( *i_id->asString(), target );

   // Not found?
   if( ! success )
   {
      // should we initialize the data?
      if ( i_func != 0 )
      {
         ClassWOPI* clw = static_cast<ClassWOPI*>(methodOf());
         ctx->pushCode( *clw->m_stepAfterPersist );
         ctx->callItem(*i_func);
      }
      else
      {
         throw FALCON_SIGN_ERROR( PersistError, FALCON_ERROR_WOPI_PERSIST_NOT_FOUND);
      }
   }
   // otherwise, all ok
   else {
      ctx->returnFrame(target);
   }
}


/*#
   @method setPersist Wopi
   @brief Creates, updates or remove local per-thread persistent data.
   @param id Unique ID for persistent data.
   @param item A data to be stored persistently.

   This method saves O/S context-specific data, to be
   retrieved, eventually by another VM process at a later
   execution under the same O/S context (usually, a thread or a process).

   Persistent data is identified by an unique ID. An application
   can present different persistent data naming them differently.

   @note The id can be any a valid string, including an empty string.
   So, "" can be used as a valid persistent key.

   Setting the item to nil effectively removes the entry from the
   persistent storage. However, the data will be reclaimed (and finalized)
   during the next garbage collection loop.

   See @a wopi_pdata for further details.
*/
FALCON_DECLARE_FUNCTION(setPersist, "id:S,item:X")
FALCON_DEFINE_FUNCTION_P1(setPersist)
{
   // Get name of the file.
   Falcon::Item *i_id = ctx->param( 0 );
   Falcon::Item *i_item = ctx->param( 1 );

   // parameter sanity check.
   if ( i_id == 0 || ! i_id->isString() || i_item == 0 )
   {
      throw paramError();
   }

   Wopi* wopi = static_cast<Wopi*>(ctx->self().asInst());
   const String& id = *i_id->asString();

   if( i_item->isNil() )
   {
      wopi->removeContextData( id );
   }
   else {
      wopi->setContextData( id, *i_item );
   }

   ctx->returnFrame();
}

/*#
   @method sendTemplate Wopi
   @brief Configures template file and possibly sends it to the remote end.
   @param stream A Falcon stream opened for reading (or a memory string stream).
   @optparam tpd Data for template conversion.
   @optparam inMemory Work in memory and return the result instead sending it.
   @return The configured contents of the file if @b inMemory is true.
   @raise IoError on error reading the file.

   This function reads a text as-is (in binary mode) and flushes its
   contents to the remote side stream.

   If a dictionary is set as template conversion data, the data in the file
   is converted so that strings between a pair of '%' symbols are expanded in
   the text corresponding to the key in the dictionary. In example, if this is
   a template file:
   @code
      My name is %name%, pleased to meet you!
   @endcode

   The @b %name% configurable text may be changed into "John Smith" through
   the following call:
   @code
      sendTemplate( InputStream("mytemplate.txt"), ["name" => "John Smith"] )
   @endcode

   If a configurable text is not found in the @b tpd dictionary, it is removed.
   The special sequence '%%' may be used to write a single '%'.

   @note Maximum length of template configurable strings is 64.
*/
FALCON_DECLARE_FUNCTION(sendTemplate, "stream:Stream,tpd:[D],inMemory:[B]")
FALCON_DEFINE_FUNCTION_P1(sendTemplate)
{
   static Class* streamClass = Engine::instance()->stdHandlers()->streamClass();
   // Get name of the file.
   Falcon::Item *i_file = ctx->param( 0 );
   Falcon::Item *i_dict = ctx->param( 1 );
   Falcon::Item *i_inMemory = ctx->param( 2 );

   // parameter sanity check.
   if ( i_file == 0 || ! i_file->isInstanceOf( streamClass ) ||
      ( i_dict != 0 && ! ( i_dict->isDict() || i_dict->isNil() ) )
   )
   {
      throw paramError();
   }

   bool bWorkInMem = i_inMemory == 0 ? false : i_inMemory->isTrue();

   Falcon::TextWriter *outStream;
   if ( bWorkInMem )
   {
      Stream* stream = new Falcon::StringStream;
      outStream = new Falcon::TextWriter( stream );
      stream->decref();
   }
   else {
      outStream = ctx->process()->textOut();
   }

   Falcon::CoreDict *dataDict = i_dict == 0 || i_dict->isNil() ? 0 : i_dict->asDict();

   Falcon::Stream *inStream = (Falcon::Stream *)i_file->asObject()->getUserData();
   Falcon::String text(1024+96); // spare a bit of space for extra templating.
   while ( ! inStream->eof() )
   {
      if ( ! inStream->readString( text, 1024 ) )
      {
         delete outStream;
         throw new Falcon::IoError( Falcon::ErrorParam( Falcon::e_io_error, __LINE__ ).
               sysError( (uint32) inStream->lastError() ) );
      }

      // scan for templating
      if ( dataDict != 0 )
      {
         Falcon::uint32 pos0=0, pos1, pos2;
         pos1 = text.find( "%" );
         while ( pos1 != Falcon::String::npos )
         {
            // verify the other position is within 64 chars
            pos2 = text.find( "%", pos1+1);

            // did we broke the limit of the read data?
            while ( pos2 == Falcon::String::npos && text.length()-pos1 < 64 )
            {
               Falcon::uint32 c;
               if ( ! inStream->get( c ) )
               {
                  delete outStream;
                  throw new Falcon::IoError( Falcon::ErrorParam( Falcon::e_io_error, __LINE__ ).
                        sysError( (uint32) inStream->lastError() ) );
               }

               text += c;
               if ( c == '%' )
                  pos2 = text.length();
            }

            // ok; now, if we have found it, fine, else just drop it.
            if ( pos2 != Falcon::String::npos )
            {
               // write the other part of the text
               outStream->writeString( text, pos0, pos1 );
               if ( pos2 == pos1 + 1 )
                  outStream->writeString( "%" );
               else
               {
                  // find the key.
                  Falcon::String key = text.subString( pos1+1, pos2 );
                  Falcon::Item *i_value = dataDict->find( &key );
                  // write something only if found
                  if ( i_value != 0 )
                  {
                     if ( i_value->isString() )
                     {
                        outStream->writeString( *i_value->asString() );
                     }
                     else {
                        Falcon::String temp;
                        vm->itemToString( temp, i_value );
                        outStream->writeString( temp );
                     }
                  }
               }
               // search next variable
               pos0 = pos2 + 1;
               pos1 = text.find( "%", pos0 );
            }
            else {
               // just write everything that's left.
               outStream->writeString( text, pos0 );
               pos1 = pos2; // will exit loop
            }
         }

         // write the last part
         outStream->writeString( text, pos0 );
      }
      // if not using the dictionary
      else {
         outStream->writeString( text );
      }
   }

   if ( bWorkInMem )
   {
      Falcon::CoreString *gs = new Falcon::CoreString;
      static_cast<Falcon::StringStream *>(outStream)->closeToString(*gs);
      vm->retval( gs );
      delete outStream;
   }
   else
   {
      outStream->flush();
   }
}

/*#
   @method parseQuery Wopi
   @brief Explodes a query string into a dictionary.
   @param qstring A string in query string format to be parsed.
   @return A dictionary containing the keys and values in the query string.
 
*/

FALCON_FUNC Wopi_parseQuery( Falcon::VMachine *vm )
{
   Falcon::Item *i_qstring = vm->param(0);
   Falcon::Item *i_output = vm->param(1);

   // return an empty string if nil was given as parameter.
   if ( i_qstring == 0 || ! i_qstring->isString() )
   {
       throw new Falcon::ParamError(
         Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ ).
         extra( "S" ) );
   }

   LinearDict* dict = new LinearDict;
   Utils::parseQuery( *i_qstring->asString(), *dict );
   vm->retval(new CoreDict(dict));
}

/*#
   @method makeQuery Wopi
   @brief Implodes a dictionary of key/value pairs into a query string.
   @param dict A dictionary containing the data to be transformed into query.
   @return A valid query string.
 
   @note The items in the dictionary are not stringified, they are just turned
    into strings with the basic toString() method. Object.toString() overrides
    won't be respected.
*/
FALCON_FUNC Wopi_makeQuery( Falcon::VMachine *vm )
{
   Falcon::Item *i_dict = vm->param(0);
   
   // return an empty string if nil was given as parameter.
   if ( i_dict == 0 || ! i_dict->isDict() )
   {
       throw new Falcon::ParamError(
         Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ ).
         extra( "D" ) );
   }

   CoreString* cs = new CoreString();
   vm->retval( cs );
   
   CoreDict* dict = i_dict->asDict();
   ItemDict& idict = dict->items();
   Iterator iter(&idict);
   bool bFilled = false;
   while( iter.hasCurrent() )
   {
      if (bFilled )
      {
         cs->append('&');
      }
      const Item& key = iter.getCurrentKey();
      const Item& value = iter.getCurrent();
      String skey; key.toString( skey );
      String svalue; value.toString( svalue );
      
      cs->append( URI::URLEncode( skey ) );
      cs->append( '=' );
      cs->append( URI::URLEncode( svalue ) );
      
      iter.next();
      bFilled = true;
   }
}

/*#
   @function htmlEscape
   @brief Escapes a string converting HTML special characters.
   @param string The string to be converted.
   @optparam output A stream where to place the output.
   @return A converted string, or nil if writing to a string.

   This function converts special HTML characters to HTML sequences:
   - '<' is converted to '&amp;lt;'
   - '>' is converted to '&amp;gt;'
   - '&' is converted to '&amp;amp;'
   - '"' is converted to '&amp;quot;'

   If an @b output stream is specified, then the output is sent there
   instead being returned in a new string. If the output of this
   function must be sent directly to output, a true value can be
   passed to send data directly to the VM output stream (usually
   linked with web server data stream). This will spare memory and
   CPU.
*/

FALCON_FUNC htmlEscape( Falcon::VMachine *vm )
{
   Falcon::Item *i_str = vm->param(0);
   Falcon::Item *i_output = vm->param(1);

   // return an empty string if nil was given as parameter.
   if ( i_str != 0 && i_str->isNil() )
   {
      vm->retval( new Falcon::CoreString );
      return;
   }

   if ( i_str == 0 || ! i_str->isString() )
   {
      throw new Falcon::ParamError(
         Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ ).
         extra( "S,[Stream|X]" ) );
   }

   if ( i_output != 0 )
   {
       if ( i_output->isObject() && i_output->asObject()->derivedFrom( "Stream" ) )
         internal_htmlEscape_stream( *i_str->asString(),
            (Falcon::Stream *) i_output->asObject()->getUserData() );
      else
         internal_htmlEscape_stream( *i_str->asString(),
            i_output->isTrue() ? vm->stdOut() : vm->stdErr() );
      return;
   }

   const Falcon::String& str = *i_str->asString();
   Falcon::CoreString* encoded = new Falcon::CoreString( str.size() );

   Falcon::uint32 len = str.length();
   for ( Falcon::uint32 i = 0; i < len; i++ )
   {
      Falcon::uint32 chr = str[i];
      switch ( chr )
      {
         case '<': encoded->append( "&lt;" ); break;
         case '>': encoded->append( "&gt;" ); break;
         case '"': encoded->append( "&quot;" ); break;
         case '&': encoded->append( "&amp;" ); break;
         default:
            encoded->append( chr );
      }
   }

   vm->retval( encoded );
}

/*#
   @class WopiError
   @brief Error generated by wopi in case of failures.
   @optparam code A numeric error code.
   @optparam description A textual description of the error code.
   @optparam extra A descriptive message explaining the error conditions.
   @from Error code, description, extra

   Possible error codes are:
   - WopiError.SessionIO - I/O error in storing or restoring a session.
   - WopiError.SessionExipred - The session is expired.
   - WopiError.AppDataStore - Cannot store/save application specific data.
   - WopiError.AppDataRestore - Cannot restore application-specific data.
   - WopiError.SessionInvalid - The session data is invalid.

*/
FALCON_DECLARE_ERROR_INSTANCE( WopiError );
FALCON_DECLARE_ERROR_CLASS_EX( WopiError, \
         addConstant("SessionIO", FALCON_ERROR_WOPI_SESS_IO );\
         addConstant("SessionExipred", FALCON_ERROR_WOPI_SESS_EXPIRED);\
         addConstant("AppDataStore", FALCON_ERROR_WOPI_APPDATA_SER);\
         addConstant("AppDataRestore", FALCON_ERROR_WOPI_APPDATA_DESER);\
         addConstant("SessionInvalid", FALCON_ERROR_WOPI_SESS_INVALID_ID);\
         )


//============================================
// Module initialization
//============================================

Falcon::Module * wopi_module_init( ObjectFactory rqf, ObjectFactory rpf,
      ext_func_t rq_init_func, ext_func_t rp_init_func  )
{
   // initialize the module
   Falcon::Module *self = new Falcon::Module();
   self->name( "WOPI" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   InitRequestClass( self, rqf, rq_init_func );
   InitReplyClass( self, rpf, rp_init_func );
   InitErrorClass( self );
   InitUploadedClass( self );
   
   
   // create a singleton instance of %Wopi class
   Falcon::Symbol *c_wopi_o = self->addSingleton( "Wopi", &Wopi_init );
   Falcon::Symbol *c_wopi = c_wopi_o->getInstance();
   c_wopi->getClassDef()->factory( CoreWopi::factory );

   self->addClassMethod( c_wopi, "getAppData", &Wopi_getAppData ).asSymbol()
      ->addParam( "app" );
   self->addClassMethod( c_wopi, "setAppData", &Wopi_setAppData ).asSymbol()
      ->addParam( "data" )->addParam( "app" );
   self->addClassMethod( c_wopi, "getPData", &Wopi_getPData ).asSymbol()
      ->addParam( "id" )->addParam("func");
   self->addClassMethod( c_wopi, "setPData", &Wopi_setPData ).asSymbol()
      ->addParam( "id" )->addParam( "func" );
   self->addClassMethod( c_wopi, "sendTemplate", &Wopi_sendTemplate ).asSymbol()
      ->addParam( "stream" )->addParam( "tpd" )->addParam( "inMemory" );
   self->addClassMethod( c_wopi, "parseQuery", &Wopi_parseQuery ).asSymbol()
      ->addParam( "qstring" );
   self->addClassMethod( c_wopi, "makeQuery", &Wopi_makeQuery ).asSymbol()
      ->addParam( "dict" );

   // Generic functions.
   self->addExtFunc( "htmlEscape", htmlEscape )
      ->addParam( "string" )->addParam( "output" );

   return self;
}

}
}

/* end of wopi_ext.cpp */
