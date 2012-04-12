/*
   FALCON - The Falcon Programming Language.
   FILE: request_ext.cpp

   Web Oriented Programming Interface

   Request class script interface
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 14 Feb 2010 13:27:55 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/wopi/request_ext.h>
#include <falcon/wopi/request.h>
#include <falcon/wopi/reply.h>
#include <falcon/wopi/error_ext.h>
#include <falcon/wopi/session_manager.h>
#include <falcon/wopi/utils.h>
#include <falcon/time_sys.h>

/*#
   @beginmodule WOPI
*/


namespace Falcon {
namespace WOPI {

/*#
   @object Request
   @brief Main web server interface object.

   Object Request contains the informations that are retreived
   from the web server, and allows to exchange data with that.

   In forms and gets,  If the field name in the request ends with
      "[]", then the entry in the gets dictionary is an array containing all the
      values posted under the same field name

   @prop request_time Number of microseconds since 00:00:00 january 1,
      1970 UTC. Format int64.

   @prop method Original request method. Can be "GET", "POST",
      or HTTP methods.

   @prop bytes_sent Body byte count, for easy access. Format int64

   @prop content_type The Content-Type for the current request.
   @prop content_encoding Encoding through which the data was received.
   @prop content_length Full length of uploaded data, including MIME multipart headers.
         Will be -1 if unknown, and 0 if the request has only an HTTP request header
         and no body data.

   @prop user If an apache authentication check was made, this gets set to the user name.
   @prop ap_auth_type If an apache authentication check was made, this gets set to the auth type.

   @prop uri The complete URI as it was sent in the request (including the query elements).
   @prop location The portion of the URI indicating the "location" of the desired file or directory.
   @prop parsed_uri The uri, already stored in a URI class instance.
   @prop filename The filename on disk corresponding to this response.
   @prop path_info The PATH_INFO extracted from this request.
   @prop args The QUERY_ARGS extracted from this request.
   @prop remote_ip  The IP address where the request has originated.
   @prop sidField The name of the field used to carry the Session ID.
   @prop provider Name of the subsystem that is providing this WOPI interface.
   @prop autoSession Set to true (default) to have the request ID field automatically
         written in the cookies of the reply when getSession() is invoked. To manage
         manually the session cookie (or send the SID field via post/get forwarding)
         set this to false.

   @prop headers Original request headers (in a dictionary).
   @prop cookies Dictionary containing the cookies set in the request.
*/

/*#
   @property startedAt Request
   @brief Thehe value of seconds() when the script was started.

   This method returns a relative time taken when the web server integration system
   gives control to Falcon, before that the script is actually loaded and started.

   This formula:
   @code
      elapsed = seconds() - Request.startedAt
   @endcode

   gives the time elapsed in processing the script up to that line, including the
   time to setup the VM and start the execution.
*/


/*#
   @property gets Request
   @brief Fields received in the GET request method.

   If the current script is invoked by a query containing query fields in the
   URI, this property contains the a dictionary with the paris
   of key/values contained in the query. Fields whose name end
   with "[]" are translated into arrays and their values is stored
   in the order tey are found in the query.

   In example, if the page is loaded through a form containing the following
   fields:
   @code
     <form action="myscript.fal" method="GET">
     <br/>User id: <input type="text" name="id"/>
     <br/>Hobby: <input type="text" name="hobbies[ ]"/>
     <br/>Hobby: <input type="text" name="hobbies[]"/>
     <br/>Hobby: <input type="text" name="hobbies[]"/>
     </form>
   @endcode

   myscript.fal will receive the following fields in gets:
   @code
      > Request.gets["id"]   // will be the user id
      inspect( Request.gets["hobbies"] )  // will be an array
   @endcode

   Get fields can be generated directly through a query. A link
   to a falcon script followed by "?" and an URL encode query
   will be translated into a GET request, and @b Request.gets fields
   will receive the specified values.

   If a web page contains the following code:
   @code
   <a href="myscript.fal?id=my_user_id&hobbies[]=diving&hobbies[]=collections">
   @endcode
   then, myscript.fal will receive the "id" value and the array specified by hobbies
   in the "hobbies" key of the @b Request.gets property.
*/

/*#
   @property posts Request
   @brief Fields received in the POST method.

   If the current script is invoked through a form declared as having
   a post method, it will receive the values of the form fields.
   Fields whose name end
   with "[]" are translated into arrays and their values is stored
   in the order tey are found in the query.

   In example, if the page is loaded through a form containing the following
   fields:
   @code
     <form action="myscript.fal" method="POST">
     <br/>User id: <input type="text" name="id"/>
     <br/>Hobby: <input type="text" name="hobbies[]"/>
     <br/>Hobby: <input type="text" name="hobbies[]"/>
     <br/>Hobby: <input type="text" name="hobbies[]"/>
     </form>
   @endcode

   myscript.fal will receive the following fields in gets:
   @code
      > Request.posts["id"]   // will be the user id
      inspect( Request.posts["hobbies"] )  // will be an array
   @endcode

   A script may receive both @b gets and @b posts fields if the

*/

/*#
   @method getField Request
   @brief Retreives a query field from either @b Request.gets or @b Request.posts.
   @param field The field name to be found.
   @optparam defval Default value to be returned if the field is not found.
   @return A cookie, POST or GET field value (as a string).
   @raise AccessError, if no default value is given and field is not found.

   In certain cases, it is useful to retreive a query field no matter if
   it comes from a cookie, the POST part or the GET part of the query. This method
   searches first the @b Request.gets, then @b Request.posts fields and finally
   the @b Request.cookies. If the field is not found, the given @b default value is
   returned; if that parameter is not specified and the field is not found, an
   AccessError is raised.
*/

FALCON_FUNC Request_getfield( VMachine *vm )
{
   Item *i_key = vm->param( 0 );
   if ( i_key == 0 || ! i_key->isString() )
   {
      throw new ParamError(
         ErrorParam( e_inv_params, __LINE__ ).extra( "S,[X]" ) );
      return;
   }

   CoreRequest *self = dyncast<CoreRequest*>( vm->self().asObject() );
   Item res;
   if( ! self->base()->getField( *i_key->asString(), vm->regA() ) )
   {
      // should we raise or return null on error?
      bool bRaise = vm->paramCount() == 1;

      // nothing; should we raise something?
      if ( bRaise )
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__ )
            .extra( "getField" ) );
         return;
      }

      //else, return the default parameter.
      vm->retval( *vm->param(1) );
   }
   // else, the result is already in regA
}

/*#
   @method fwdGet Request
   @brief Forwards the request, creating a suitable query string for a target URI.
   @optparam all If true, add also the POST fields.
   @return An URI encoded string that can be directly used in further get requests.

   This method simplifies the task to create callback to the page that is being
   processed when it's necessary to forward all the fields received to the new request.

   If the @b all parameter is true, also the fields passed as POST fields will be
   forwared through this method.

   @note As the get and post fields are not read-only, it is possible to change their
   contents in this object and then call this method to introduce exceptions in forwarding
   the request.
*/

FALCON_FUNC Request_fwdGet( VMachine *vm )
{
   CoreRequest *self = dyncast<CoreRequest*>( vm->self().asObject() );
   Item *i_all = vm->param( 0 );

   CoreString *res = new CoreString;

   self->base()->fwdGet( *res, i_all != 0 && i_all->isTrue() );
   vm->retval( res );
}

/*#
   @method fwdPost Request
   @brief Forwards the request, creating a set of hidden input fields.
   @optparam all If true, add also the GET fields.
   @return A string containing pre-encoded http hidden input fields.

   This method simplifies the task to create callback to the page that is being
   processed when it's necessary to forward all the fields received to the new request.

   If the @b all parameter is true, also the fields passed as GET fields will be
   forwared through this method.

   @note As the get and post fields are not read-only, it is possible to change their
   contents in this object and then call this method to introduce exceptions in forwarding
   the request.
*/

FALCON_FUNC Request_fwdPost( VMachine *vm )
{
   CoreRequest *self = dyncast<CoreRequest*>( vm->self().asObject() );
   Item *i_all = vm->param( 0 );

   CoreString *res = new CoreString;

   self->base()->fwdPost( *res, i_all != 0 && i_all->isTrue() );
   vm->retval( res );
}

/*#
   @method getSession Request
   @brief Create a new session or returns the session data associated with this session.
   @optparam sid Explicit session ID synthesized by the script.
   @return A blessed dictionary that can be used to store session data.

   @raise WopiError If the session cannot be restored or is expired. Use the return code
      to determine what happened.

   This method creates a new session, eventually using an externally provided
   session ID. If not provided, the session ID is found in cookies or other
   sources as indicated by the auto-session settings.

   If a @b sid is provided, the owner is responsible for its creation and maintenance.

   Possible errors that can be thrown are:
    - WopiError.SessionFailed - Failed to restore a session.
    - WopiError.SessionInvalid - Invalid session ID.

*/

FALCON_FUNC Request_getSession( VMachine *vm )
{
   CoreRequest *self = dyncast<CoreRequest*>( vm->self().asObject() );
   Item *i_sid = vm->param( 0 );
   if ( i_sid != 0 && ! i_sid->isString() )
   {
      throw new ParamError(
         ErrorParam( e_inv_params, __LINE__ ).extra( "[S]" ) );
      return;
   }

   // Get the field name
   const String& fieldName = self->base()->getSessionFieldName();

   Item i_sid_v;
   bool success;
   if( i_sid != 0 )
   {
	   i_sid_v = *i_sid;
	   success = true;
   }
   else
   {
	   success = self->base()->getField( fieldName, i_sid_v ) && i_sid_v.isString();
   }

   // Get the session id, if it's there.
   WOPI::SessionData* sd = 0;
   if( success && *i_sid_v.asString() != "" )
   {
      sd = self->smgr()->getSession( *i_sid_v.asString(), self->base()->sessionToken() );
      if( sd == 0 )
      {
         throw new WopiError( ErrorParam( FALCON_ERROR_WOPI_SESS_EXPIRED, __LINE__ )
                        .desc( "Invalid or unexisting session" )
                        .extra("getSession:" + *i_sid_v.asString() ) );
      }
      else if ( sd->isInvalid() )
      {
         String desc = sd->errorDesc();
         delete sd;

         // eventually kill the session id
         if( self->autoSession() )
         {
            self->reply()->setCookie( fieldName, CookieParams() );
         }

         throw new WopiError( ErrorParam( FALCON_ERROR_WOPI_SESS_IO, __LINE__ )
               .desc( "Failed to restore the required session" )
               .extra( desc ) );

      }
   }
   else
   {
      if( i_sid != 0 )
      {
         sd = self->smgr()->startSession( self->base()->sessionToken(), *i_sid->asString() );
         if( sd == 0 )
         {
            throw new WopiError( ErrorParam( FALCON_ERROR_WOPI_SESS_INVALID_ID, __LINE__ )
                  .desc( "Duplicated Session ID"));
         }
      }
      else
      {
         sd = self->smgr()->startSession( self->base()->sessionToken() );
      }
   }

   // ok; now, should we replicate the session on the cookies?
   if( self->autoSession() && i_sid == 0 )
   {
      TimeStamp expirets;
      CookieParams cpar;
      cpar.value( sd->sID() );            
         
      if ( self->smgr()->timeout() > 0 )
      {
         expirets.currentTime();
         expirets.changeTimezone(tz_UTC);
         expirets.add(0,0,0,self->smgr()->timeout());
         cpar.expire( &expirets );
      }
      
      self->reply()->setCookie( fieldName, cpar );
   }
   vm->retval( sd->data() );
}

/*#
   @method startSession Request
   @brief Create a new session.
   @optparam sid Explicit session ID synthesized by the script.
   @return A blessed dictionary that can be used to store session data.
   @raise WopiError If the session cannot be created. Use the return code
      to determine what happened.

   This method creates a new session, using an explicit session ID. The session ID
   is not automatically saved in cookies nor propagated by any other means.

   Possible errors that can be thrown are:
    - WopiError.SessionFailed - Failed create the session (session already existing)

*/

FALCON_FUNC Request_startSession( VMachine *vm )
{
   CoreRequest *self = dyncast<CoreRequest*>( vm->self().asObject() );
   Item *i_sid = vm->param( 0 );
   if ( i_sid == 0 || ! i_sid->isString() )
   {
      throw new ParamError(
         ErrorParam( e_inv_params, __LINE__ ).extra( "[S]" ) );
      return;
   }

   // Get the field name
   SessionData* sd = self->smgr()->startSession( self->base()->sessionToken(), *i_sid->asString() );
   if( sd == 0 )
   {
      throw new WopiError( ErrorParam( FALCON_ERROR_WOPI_SESS_INVALID_ID, __LINE__ )
            .desc( "Duplicated Session ID"));
   }

   vm->retval( sd->data() );
}

/*#
   @method closeSession Request
   @optparam sid Optional explicit SID to be closed
   @brief Closes a currently open session.
   
   If the current script is associated with an open session, the active session
   is closed. In case the session control keeps track of the session ID using
   a cookie, this cookie is automatically removed.
   
   In case the current script is not associated with a session, this method does
   nothing.
*/

FALCON_FUNC Request_closeSession( VMachine *vm )
{
   CoreRequest *self = dyncast<CoreRequest*>( vm->self().asObject() );

   Item *i_sid_param = vm->param( 0 );
   if ( i_sid_param != 0 && ! i_sid_param->isString() )
   {
      throw new ParamError(
         ErrorParam( e_inv_params, __LINE__ ).extra( "[S]" ) );
      return;
   }

   // explicit SID to be closed?
   if ( i_sid_param != 0 )
   {
      self->smgr()->closeSession( *i_sid_param->asString(), self->base()->sessionToken() );
      // ok, we're done
      return;
   }

   // Get the field name
   const String& fieldName = self->base()->getSessionFieldName();
   
   Item i_sid;
   if( self->base()->getField( fieldName, i_sid ) && i_sid.isString() && *i_sid.asString() != "" )
   {
      self->smgr()->closeSession( *i_sid.asString(), self->base()->sessionToken() );
      
      // ok; In case of autosession, close the cookie.
      if( self->autoSession() )
      {
         self->reply()->clearCookie( fieldName );
         self->base()->cookies()->remove( fieldName );
      }
   }
}

/*#
   @method hasSession Request
   @brief Checks if the current script is hosting an open session.
   @return True if this script supposes that it has an open session.
   
   This function checks this script has been provided with a seemingly valid
   session id; in other words, it checks if @a Request.getSession() will try to open
   an existing session or if it will create a new session.
   
   @a Request.getSession() may fail even if this function returns true, in case
   the session ID that this script is provided with is invalid or expired, or if
   an I/O error occurs during session restore. 
*/

FALCON_FUNC Request_hasSession( VMachine *vm )
{

   CoreRequest *self = dyncast<CoreRequest*>( vm->self().asObject() );

   // Get the field name
   const String& fieldName = self->base()->getSessionFieldName();
   
   Item i_sid;
   bool res = self->base()->getField( fieldName, i_sid ) && i_sid.isString() && *i_sid.asString() != "";
   vm->regA().setBoolean(res);
}


/*#
   @method tempFile Request
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
FALCON_FUNC Request_tempFile( Falcon::VMachine *vm )
{
   CoreRequest* request = Falcon::dyncast<CoreRequest*>(vm->self().asObject());

   String fname;
   int64 le;
   Stream* tgFile = request->base()->makeTempFile( fname, le );
   if( tgFile == 0 )
   {
      throw new Falcon::IoError( Falcon::ErrorParam( Falcon::e_io_error, __LINE__ )
         .extra( fname )
         .sysError( (uint32) le ) );
   }

   Item* i_name = vm->param(0);

   // create the stream.
   Falcon::Item *stream_cls = vm->findWKI( "Stream" );
   fassert( stream_cls != 0 );
   fassert( stream_cls->isClass() );
   Falcon::CoreObject *oret = stream_cls->asClass()->createInstance();
   oret->setUserData( tgFile );
   vm->retval( oret );

   // eventually report the filename
   if ( i_name != 0 && vm->isParamByRef(0) )
   {
      Falcon::CoreString* gs = new Falcon::CoreString( fname );
      gs->bufferize();
      *i_name = gs;
   }
}


void InitRequestClass( Module* self, ObjectFactory cff, ext_func_t init_func )
{
   // create a singleton instance of %Request class
   Falcon::Symbol *c_request_o = self->addSingleton( "Request", init_func );
   Falcon::Symbol *c_request = c_request_o->getInstance();
   c_request->getClassDef()->factory( cff );


   self->addClassProperty( c_request, "gets" );
   self->addClassProperty( c_request, "posts" );
   self->addClassProperty( c_request, "cookies" );
   self->addClassProperty( c_request, "headers" );
   self->addClassProperty( c_request, "parsed_uri" );
   self->addClassProperty( c_request, "autoSession" );

   // Fields from apache request_rec
   self->addClassProperty( c_request, "protocol" );
   self->addClassProperty( c_request, "request_time" );
   self->addClassProperty( c_request, "method" );
   self->addClassProperty( c_request, "bytes_sent" );
   self->addClassProperty( c_request, "content_type" );
   self->addClassProperty( c_request, "content_encoding" );
   self->addClassProperty( c_request, "content_length" );
   self->addClassProperty( c_request, "ap_auth_type" );
   self->addClassProperty( c_request, "user" );
   self->addClassProperty( c_request, "uri" );
   self->addClassProperty( c_request, "location" );
   self->addClassProperty( c_request, "filename" );
   self->addClassProperty( c_request, "path_info" );
   self->addClassProperty( c_request, "args" );
   self->addClassProperty( c_request, "remote_ip" );
   self->addClassProperty( c_request, "sidField" );
   self->addClassProperty( c_request, "startedAt" );
   self->addClassProperty( c_request, "provider" );

   self->addClassMethod( c_request, "getField", &Request_getfield ).asSymbol()
      ->addParam( "field" )->addParam( "defval" );
   self->addClassMethod( c_request, "fwdGet", &Request_fwdGet ).asSymbol()
      ->addParam( "all" );
   self->addClassMethod( c_request, "fwdPost", &Request_fwdPost ).asSymbol()
      ->addParam( "all" );
   self->addClassMethod( c_request, "startSession", &Request_startSession ).asSymbol()
      ->addParam("sid");
   self->addClassMethod( c_request, "getSession", &Request_getSession ).asSymbol()
      ->addParam("sid");
   self->addClassMethod( c_request, "closeSession", &Request_closeSession );
   self->addClassMethod( c_request, "hasSession", &Request_hasSession );
   self->addClassMethod( c_request, "tempFile", &Request_tempFile ).asSymbol()
      ->addParam( "name" );
}

}
}

/* end of request_ext.cpp */
