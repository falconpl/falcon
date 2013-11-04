/*
   FALCON - The Falcon Programming Language.
   FILE: classrequest.cpp

   Falcon Web Oriented Programming Interface.

   Interface to Request object
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 16 Oct 2013 12:38:19 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/wopi/classrequest.h>
#include <falcon/wopi/request.h>
#include <falcon/wopi/errors.h>
#include <falcon/vmcontext.h>


/*#
   @beginmodule WOPI
*/


namespace Falcon {
namespace WOPI {

/*#
   @object Request
   @brief Main web server interface object.

   Object Request contains the informations that are retrieved
   from the web server, and allows to exchange data with that.

   In forms and gets,  If the field name in the request ends with
      "[]", then the entry in the gets dictionary is an array containing all the
      values posted under the same field name

   @prop request_time Number of microseconds since 00:00:00 January 1,
      1970 UTC. Format int64.

   @prop method Original request method. Can be "GET", "POST",
      or HTTP methods.

   @prop bytes_sent Body byte count, for easy access. Format int64

   @prop content_type The Content-Type for the current request.
   @prop content_encoding Encoding through which the data was received.
   @prop content_length Full length of uploaded data, including MIME multipart headers.
         Will be -1 if unknown, and 0 if the request has only an HTTP request header
         and no body data.

   @prop user If an Apache authentication check was made, this gets set to the user name.
   @prop ap_auth_type If an Apache authentication check was made, this gets set to the auth type.

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

namespace {

static void get_gets( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   value.setUser( request->gets()->handler(), request->gets() );
}

static void get_posts( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   value.setUser( request->posts()->handler(), request->posts() );
}

static void get_cookies( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   value.setUser( request->cookies()->handler(), request->cookies() );
}

static void get_headers( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   value.setUser( request->posts()->handler(), request->posts() );
}

static void get_parserd_uri( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   const String& uri = request->parsedUri().encode();
   value = FALCON_GC_HANDLE( new String( uri ) );
}

static void get_protocol( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   value = FALCON_GC_HANDLE( new String( request->m_protocol ) );
}

static void get_request_time( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   value = request->m_request_time;
}

static void get_bytes_sent( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   value = (int64) request->m_bytes_received;
}

static void get_content_length( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   value = (int64) request->m_content_length;
}

static void get_method( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   value = (int64) request->m_content_length;
}

static void get_content_type( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   value = FALCON_GC_HANDLE( new String( request->m_content_type ) );
}

static void get_content_encoding( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   value = FALCON_GC_HANDLE( new String( request->m_content_encoding ) );
}

static void get_ap_auth_type( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   value = FALCON_GC_HANDLE( new String( request->m_ap_auth_type ) );
}

static void get_user( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   value = FALCON_GC_HANDLE( new String( request->m_user ) );
}

static void get_location( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   value = FALCON_GC_HANDLE( new String( request->m_location ) );
}

static void get_uri( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   value = FALCON_GC_HANDLE( new String( request->m_uri.encode() ) );
}

static void get_filename( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   value = FALCON_GC_HANDLE( new String( request->m_filename ) );
}

static void get_path_info( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   value = FALCON_GC_HANDLE( new String( request->m_path_info ) );
}

static void get_args( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   value = FALCON_GC_HANDLE( new String( request->m_args ) );
}

static void get_remote_ip( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   value = FALCON_GC_HANDLE( new String( request->m_remote_ip ) );
}

static void get_startedAt( const Class*, const String&, void* inst, Item& value )
{
   Request* request = static_cast<Request*>(inst);
   value = request->startedAt();
}

/*#
   @method getField Request
   @brief Retrieves a query field from either @b Request.gets or @b Request.posts.
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

FALCON_DECLARE_FUNCTION( getField, "field:S,defval:[X]")
FALCON_DEFINE_FUNCTION_P1( getField )
{
   Item *i_key = ctx->param( 0 );
   if ( i_key == 0 || ! i_key->isString() )
   {
      throw paramError();
   }

   Request *self = ctx->tself<Request*>();

   Item res;
   const String& fieldName = *i_key->asString();
   if( ! self->getField( fieldName,res ) )
   {
      // should we raise or return null on error?
      bool bRaise = ctx->paramCount() == 1;

      // nothing; should we raise something?
      if ( bRaise )
      {
         throw FALCON_SIGN_XERROR( AccessError, FALCON_ERROR_WOPI_FIELD_NOT_FOUND,
            .extra( fieldName ) );
      }
      // else, set res as the default value
      fassert( ctx->param(1) != 0 );
      res = *ctx->param(1);
   }

   ctx->returnFrame(res);
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

FALCON_DECLARE_FUNCTION( fwdGet, "all:[B]")
FALCON_DEFINE_FUNCTION_P1( fwdGet )
{
   Request *self = ctx->tself<Request*>();
   Item *i_all = ctx->param( 0 );
   String* res = new String;
   self->fwdGet( *res, i_all != 0 && i_all->isTrue() );
   ctx->returnFrame( FALCON_GC_HANDLE(res) );
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

FALCON_DECLARE_FUNCTION( fwdPost, "all:[B]")
FALCON_DEFINE_FUNCTION_P1( fwdPost )
{
   Request *self = ctx->tself<Request*>();
   Item *i_all = ctx->param( 0 );
   String* res = new String;
   self->fwdPost( *res, i_all != 0 && i_all->isTrue() );
   ctx->returnFrame( FALCON_GC_HANDLE(res) );
}


}

//============================================================================================
// Request class
//

ClassRequest::ClassRequest():
         Class("%Request")
{
   addProperty( FALCON_WOPI_REQUEST_GETS_PROP, &get_gets );
   addProperty( FALCON_WOPI_REQUEST_POSTS_PROP, &get_posts );
   addProperty( FALCON_WOPI_REQUEST_COOKIES_PROP, &get_cookies );
   addProperty( FALCON_WOPI_REQUEST_HEADERS_PROP, &get_headers );
   addProperty( FALCON_WOPI_REQUEST_PARSED_URI_PROP, &get_parserd_uri );
   addProperty( FALCON_WOPI_REQUEST_PROTOCOL_PROP, &get_protocol );
   addProperty( FALCON_WOPI_REQUEST_REQUEST_TIME_PROP, &get_request_time );
   addProperty( FALCON_WOPI_REQUEST_BYTES_SENT_PROP, &get_bytes_sent );
   addProperty( FALCON_WOPI_REQUEST_CONTENT_LENGHT_PROP, &get_content_length );
   addProperty( FALCON_WOPI_REQUEST_METHOD_PROP, &get_method );
   addProperty( FALCON_WOPI_REQUEST_CONTENT_TYPE_PROP, &get_content_type );
   addProperty( FALCON_WOPI_REQUEST_CONTENT_ENCODING_PROP, &get_content_encoding );
   addProperty( FALCON_WOPI_REQUEST_AP_AUTH_TYPE_PROP, &get_ap_auth_type );
   addProperty( FALCON_WOPI_REQUEST_USER_PROP, &get_user );
   addProperty( FALCON_WOPI_LOCATION_PROP, &get_location );
   addProperty( FALCON_WOPI_URI_PROP, &get_uri );
   addProperty( FALCON_WOPI_FILENAME_PROP, &get_filename );
   addProperty( FALCON_WOPI_PATH_INFO_PROP, &get_path_info );
   addProperty( FALCON_WOPI_ARGS_PROP, &get_args );
   addProperty( FALCON_WOPI_REMOTE_IP_PROP, &get_remote_ip );
   addProperty( FALCON_WOPI_STARTED_AT_PROP, &get_startedAt );


   addMethod( new Function_getField );
   addMethod( new Function_fwdPost );
   addMethod( new Function_fwdGet );
}


ClassRequest::~ClassRequest()
{
}


void ClassRequest::dispose( void* ) const
{
   // do nothing
}


void* ClassRequest::clone( void* ) const
{
   // uncloneable.
   return 0;
}


void* ClassRequest::createInstance() const
{
   // abstract class
   return 0;
}

void ClassRequest::gcMarkInstance( void* instance, uint32 mark ) const
{
   Request* r = static_cast<Request*>(instance);
   r->gcMark(mark);
}

bool ClassRequest::gcCheckInstance( void* instance, uint32 mark ) const
{
   Request* r = static_cast<Request*>(instance);
   return r->currentMark() >= mark;
}


}
}

/* end of classrequest.cpp */
