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

#include <falcon/wopi/version.h>

#include <falcon/wopi/wopi.h>
#include <falcon/wopi/modulewopi.h>
#include <falcon/wopi/classwopi.h>
#include <falcon/wopi/errors.h>
#include <falcon/wopi/utils.h>

#include <falcon/itemdict.h>
#include <falcon/itemarray.h>
#include <falcon/vmcontext.h>
#include <falcon/function.h>
#include <falcon/textwriter.h>
#include <falcon/stringstream.h>

#include <falcon/engine.h>
#include <falcon/stdhandlers.h>
#include <stdlib.h>

/*#
   @beginmodule WOPI
*/

namespace Falcon {
namespace WOPI {

namespace {
class PStepAfterPersist: public PStep
{
public:
   inline PStepAfterPersist() { apply = apply_; }
   inline virtual ~PStepAfterPersist() {}
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


/*#
@object Wopi
@brief General configuration and WOPI-system wide interface

This object gives access to generic configuration settings and global
module functions.
*/


/*#
   @method tempFile Wopi
   @brief Creates a temporary file.
   @optparam name If given and passed by reference, it will receive the complete file name.
   @return A Falcon stream.
   @raise IoError on open or write error.

   Temporary streams are automatically deleted when the the script terminates.

   In case the script want to get the name of the file where the temporary data
   is stored (i.e. to copy or move it elsewhere after having completed the updates),
   the parameter @b name needs to be passed by reference, and it will receive the
   filename.

   @note The temporary files are stored in the directory specified by the
      parameter UploadDir in the falcon.ini file.
*/

FALCON_DECLARE_FUNCTION(tempFile, "name:[S]" )
FALCON_DEFINE_FUNCTION_P(tempFile)
{
   Wopi* wopi = ctx->tself<Wopi*>();

   String fname;
   Stream* tgFile;
   if( pCount == 0 )
   {
      tgFile = wopi->makeTempFile();
   }
   else
   {
      Item* i_fname = ctx->param(0);
      if( ! i_fname->isString() )
      {
         throw paramError();
      }

      // we need a copy of the string.
      String fname = *i_fname->asString();
      tgFile = wopi->makeTempFile(fname, false);
   }

   ctx->returnFrame( FALCON_GC_HANDLE(tgFile) );
}


/*#
   @property scriptName Wopi
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
   @property scriptPath Wopi
   @brief Physical name of the loaded main script.

   This is the name of the script that has been loaded and served through the
   web server request. It may be different from the file name that has been
   actually requested, as the web server may perform several mappings.

   scriptPath + "/" + scriptName is granted to give the complete file path
   where the script has been loaded.
*/
static void get_scriptPath(const Class* cls, const String&, void* instance, Item& value )
{
   TRACE1( "WOPI.scriptPath for %p", instance );
   ModuleWopi* modwopi = static_cast<ModuleWopi*>( cls->module() );
   value = FALCON_GC_HANDLE(new String(modwopi->scriptPath()));
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
         ClassWopi* clw = static_cast<ClassWopi*>(methodOf());
         ctx->pushCode( clw->m_stepAfterPersist );
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
   @method parseQuery Wopi
   @brief Explodes a query string into a dictionary.
   @param qstring A string in query string format to be parsed.
   @return A dictionary containing the keys and values in the query string.
 
*/

FALCON_DECLARE_FUNCTION(parseQuery, "qstring:S")
FALCON_DEFINE_FUNCTION_P1(parseQuery)
{
   Falcon::Item *i_qstring = ctx->param(0);

   // return an empty string if nil was given as parameter.
   if ( i_qstring == 0 || ! i_qstring->isString() )
   {
       throw paramError();
   }

   ItemDict* dict = new ItemDict;
   Utils::parseQuery( *i_qstring->asString(), *dict );
   ctx->returnFrame(FALCON_GC_HANDLE(dict));
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
FALCON_DECLARE_FUNCTION(makeQuery, "dict:D")
FALCON_DEFINE_FUNCTION_P1(makeQuery)
{
   Falcon::Item *i_dict = ctx->param(0);
   
   // return an empty string if nil was given as parameter.
   if ( i_dict == 0 || ! i_dict->isDict() )
   {
       throw paramError();
   }
   

   ItemDict* idict = i_dict->asDict();
   String* str = new String;

   class Rator: public ItemDict::Enumerator
   {
   public:
      Rator( String& tgt ): m_tgt(tgt){}
      virtual ~Rator() {}
      virtual void operator()( const Item& key, Item& value )
      {
         if( !m_tgt.empty() )
         {
            m_tgt.append('&');
         }

         String sKey, sValue;
         key.describe(sKey);
         value.describe(sValue);
         m_tgt.append( URI::URLEncode( sKey ) );
         m_tgt.append( '=' );
         m_tgt.append( URI::URLEncode( sValue ) );
      }
      
   private:
      String& m_tgt;
   }
   rator( *str );

   idict->enumerate(rator);
   ctx->returnFrame( FALCON_GC_HANDLE(idict) );
}

/*#
   @method escape Wopi
   @brief Escapes a string converting HTML special characters.
   @param string The string to be converted.
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

FALCON_DECLARE_FUNCTION(escape, "string:S")
FALCON_DEFINE_FUNCTION_P1(escape)
{
   Falcon::Item *i_str = ctx->param(0);

   // return an empty string if nil was given as parameter.
   if ( i_str != 0 && i_str->isNil() )
   {
      ctx->returnFrame( FALCON_GC_HANDLE(new String) );
      return;
   }

   if ( i_str == 0 || ! i_str->isString() )
   {
      throw paramError();
   }


   const Falcon::String& str = *i_str->asString();
   String* encoded = new String( str.size() );

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
            break;
      }
   }

   ctx->returnFrame( FALCON_GC_HANDLE(encoded) );
}

}

//===================================================================================
// Wopi class
//

ClassWopi::ClassWopi():
         Class("%Wopi")
{
   m_stepAfterPersist = new PStepAfterPersist;

   addProperty("scriptName", &get_scriptName );
   addProperty("scriptPath", &get_scriptPath );

   addMethod( new Function_tempFile );
   addMethod( new Function_escape );
   addMethod( new Function_parseQuery );
   addMethod( new Function_makeQuery );
}

ClassWopi::~ClassWopi()
{
   delete m_stepAfterPersist;
}

void ClassWopi::dispose( void* ) const
{
   // do nothing
}

void* ClassWopi::clone( void* ) const
{
   return 0;
}

void* ClassWopi::createInstance() const
{
   // static class
   return 0;
}

}
}

/* end of classwopi.cpp */
