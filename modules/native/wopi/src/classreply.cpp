/*
   FALCON - The Falcon Programming Language.
   FILE: classreply.cpp

   Web Oriented Programming Interface

   Reply class script interface
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 19 Feb 2010 22:09:02 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/wopi/classreply.h>
#include <falcon/wopi/reply.h>
#include <falcon/wopi/utils.h>
#include <falcon/wopi/modulewopi.h>

#include <falcon/itemid.h>
#include <falcon/stderrors.h>
#include <falcon/vmcontext.h>

#include <ctype.h>

/*#
   @beginmodule WOPI
*/


namespace Falcon {
namespace WOPI {

namespace {

static void internal_capitalize_name( String& name )
{
   // capitalize the first letter
   name.setCharAt( 0, toupper( name[0] ) );

   // and all the letters after a "-"
   uint32 pos = name.find( "-" );

   // we know name is long at least 1
   while( pos < name.length()-1 )
   {
      name.setCharAt( pos+1, toupper( name[pos+1] ) );
      pos = name.find( "-", pos + 1 );
   }

}


/*#
   @object Reply
   @brief Reflects the reply that this script has sent (or will send).

   This object contains the header that the system is going to send back
   to the remote client as soon as the first output (through the ">" operator
   or through print functions) is performed. If output has already been
   performed, then the header becomes read-only, and the @b Reply.sent field
   becomes true.

   Falcon template files willing to change the output header should escape
   to Falcon immediately, giving the escape sequence as their very first
   character, as unescaped text is immediately sent to the upstream server.

   A suitable default header with "+200 OK", no cache pragmas and text/html in
   utf-8 encoding is provided by default.
   
   @prop status Numeric HTTP reply code. For example, 200 (meaning OK).

   @prop reason Descriptive http reply reason. In the "+200 OK" standard reply,
      it's the "OK" part.

   @prop sent True when the header has been delivered, false if the header is
      still unsent.
*/

static void get_status(const Class*, const String&, void* instance, Item& value )
{
   Reply*r = static_cast<Reply*>(instance);
   value = (int64) r->status();
}

static void set_status(const Class*, const String&, void* instance, const Item& value )
{
   Reply*r = static_cast<Reply*>(instance);
   r->status((int32) value.asInteger());
}

static void get_reason(const Class*, const String&, void* instance, Item& value )
{
   Reply*r = static_cast<Reply*>(instance);
   value = FALCON_GC_HANDLE( new String(r->reason()) );
}

static void set_reason(const Class*, const String&, void* instance, const Item& value )
{
   Reply*r = static_cast<Reply*>(instance);
   if( ! value.isString() )
   {
      throw FALCON_SIGN_XERROR( AccessTypeError, e_inv_prop_value, .extra("S") );
   }
   r->reason( *value.asString() );
}

static void get_isSent(const Class*, const String&, void* instance, Item& value )
{
   Reply*r = static_cast<Reply*>(instance);
   value.setBoolean( r->isCommited() );
}


/*#
   @method commit Reply
   @brief Sends immediately the header and pending data.

   Usually, the integration module sends the header upstream
   at a reasonable moment, but some scripts may prefer to send
   the header sooner; in example, this is useful if the script
   is just generating a reply that consists in an HTTP header
   without data, or to start a keep-alive HTTP/1.1 conversation.
*/
FALCON_DECLARE_FUNCTION(commit, "")
FALCON_DEFINE_FUNCTION_P1(commit)
{
   Reply* r = ctx->tself<Reply*>();
   ctx->returnFrame(Item().setBoolean( r->commit() ));
}

/*#
   @method setCookie Reply
   @brief sets a cookie that will be received next time that a script is called.
   @param name Name of the Cookie or complete cookie specification in a dictionary.
   @optparam value Value for the cookie (as a string).
   @optparam expires Expiration date (a TimeStamp or an ISO or RFC2822 formatted string),
             or maximum life in seconds (an integer).
   @optparam path Cookie path validity
   @optparam domain Domain for which the cookie is valid.
   @optparam secure True to specify that the cookie can be sent only in https.
   @optparam httpOnly If true, this cookie will be invisible to client side scripts, and will
             be only sent to the server.

   This facility allows to store variables on the remote system (usually
   a web browser), which will send them back each time it connects again.

   The cookies sent through this function will be received in the @b cookies
   member of subsequent call to this or other scripts.

   Parameters to be left blank can be skipped using @b nil; however, it may be useful
   to use the named parameter convention.

   For example:
   @code
      Reply.setCookie( "cookie1", "value1", nil, nil, ".mydomain.com", false, true )

      // but probably better
      Reply.setCookie(
         name|"cookie1",
         value|"value1",
         domain|".mydomain.com",
         httpOnly|true
         )
   @endcode

   Only the @b name parameter is mandatory.
   @note Cookies must be set before output is sent to the upstream server.
*/
FALCON_DECLARE_FUNCTION(setCookie, "name:S,value:[S],expires:[N|S|TimeStamp],path:[S],domain:[S],secure:[B],httpOnly:[B]")
FALCON_DEFINE_FUNCTION_P1(setCookie)
{
   String sDummy;
   String sCookie;

   Item *i_name = ctx->param(0);
   Item *i_value = ctx->param(1);
   Item *i_expire = ctx->param(2);
   Item *i_path = ctx->param(3);
   Item *i_domain = ctx->param(4);
   Item *i_secure = ctx->param(5);
   Item *i_httpOnly = ctx->param(6);

   CookieParams cp;

   if ( i_name == 0 || ! (i_name->isString() ) )
   {
      throw paramError();
   }

   // if value is not a string, stringify it
   if ( i_value != 0 )
   {
      if ( i_value->isString() )
      {
         cp.value( *i_value->asString() );
      }
      else if( ! i_value->isNil() )
      {
         cp.value(i_value->describe());
      }
   }

   // Expire part
   if ( i_expire != 0 )
   {
      if ( i_expire->isOrdinal() )
      {
         cp.m_max_age = (int32) i_expire->forceInteger();
      }
      // a bit of sanitization; if we have an expire, we must ensure it's in ISO or RFC2822 format.
      else if ( i_expire->type() == FLC_CLASS_ID_TIMESTAMP )
      {
         cp.m_expire = (TimeStamp *) i_expire->asInst();
      }
      else if ( i_expire->isString() )
      {
         TimeStamp tsDummy;
         if ( ! TimeStamp::fromRFC2822( tsDummy, *i_expire->asString() ) )
         {
            throw paramError("Invalid RFC2822 TimeStamp");
         }

         cp.m_expire_string = *i_expire->asString();
      }
      else if ( ! i_expire->isNil() )
      {
         throw paramError();
      }
   }

   // path part
   if ( i_path != 0 )
   {
      if ( i_path->isString() )
      {
         cp.m_path = *i_path->asString();
      }
      else if ( ! i_path->isNil() )
      {
         throw paramError();
      }
   }

   if ( i_domain != 0 )
   {
      if ( i_domain->isString() )
      {
         cp.m_domain = *i_domain->asString();
      }
      else if ( ! i_domain->isNil() )
      {
         throw paramError();
      }
   }

   if ( i_secure != 0 && i_secure->isTrue() )
   {
      cp.m_secure = true;
   }

   if ( i_httpOnly != 0 && i_httpOnly->isTrue() )
   {
      cp.m_httpOnly = true;
   }

   // great, we have it.
   {
      Reply* r = ctx->tself<Reply*>();
      r->setCookie( *i_name->asString(), cp );
   }

   ctx->returnFrame();
}


/*#
   @method clearCookie Reply
   @brief Remove given cookie.
   @param name The cookie to be removed.
   
   This function explicitly tries to clear the cookie from the remote client cache.
   The cookie value is @b not removed from the @a Request.cookies array.
*/

FALCON_DECLARE_FUNCTION(clearCookie, "name:S")
FALCON_DEFINE_FUNCTION_P1(clearCookie)
{
   String sCookie;

   Item *i_name = ctx->param(0);
   if( i_name == 0 || ! i_name->isString() )
   {
      throw paramError();
   }
 
   Reply* r = ctx->tself<Reply*>();
   r->clearCookie( *i_name->asString() );
   ctx->returnFrame();
}



/*#
   @method redirect Reply
   @brief Helper creating a Refresh (redirection) header.
   @optparam uri Where to redirect the page.
   @optparam timeout Number of seconds before refresh takes place.

   This function creates a well-formed "Refresh" header in the reply, which
   can:
   - Reload the page immediately or after a timeout
   - Ask for redirection to another page immediately or after a timeout.

   @note As this method mangles the output headers, it must be called before
   any output is sent to the client. Otherwise, it will be ignored.
*/

FALCON_DECLARE_FUNCTION(redirect, "uri:[S],timeout:[N]")
FALCON_DEFINE_FUNCTION_P1(redirect)
{
   Reply* r = ctx->tself<Reply*>();

   Item* i_url = ctx->param(0);
   Item* i_timeout = ctx->param(1);

   if( (i_url != 0 && !( i_url->isNil() || i_url->isString() ))
      || (i_timeout != 0 && ! ( i_timeout->isNil() || i_timeout->isOrdinal() ))
   )
   {
      throw paramError();
   }

   int iTimeout = (int)(i_timeout == 0 || i_timeout->isNil() ? 0: i_timeout->forceInteger());

   if( i_url == 0 || i_url->isNil() )
   {
      r->setRedirect( "", iTimeout );
   }
   else
   {
      r->setRedirect( *i_url->asString(), iTimeout );
   }

   ctx->returnFrame();
}

/*#
   @method setHeader Reply
   @brief Adds or remove a reply header.
   @param name The header to be created or removed.
   @optparam value If given and not nil, the value to be stored in the header; if not given,
             or nil, the header will be removed.

   This sets the given header to the required value. The @b value parameter can be of
   any type; if it's not a string, a standard conversion to string will be attempted.

   In case @b value is not given or is nil, the header is removed.

   The header @b name is automatically capitalized at the beginning and after each '-'
   symbol.

   @note Pay attention to the fact that the maps in which headers are stored are
   case-sensitive. A case mismatch may cause an undesired duplication of the header.
*/

FALCON_DECLARE_FUNCTION(setHeader, "name:S,value:[S]")
FALCON_DEFINE_FUNCTION_P1(setHeader)
{
   Reply* r = ctx->tself<Reply*>();

   Item* i_name = ctx->param(0);
   Item* i_value = ctx->param(1);

   if( i_name == 0 || ! i_name->isString() )
   {
      throw paramError();
   }

   String name = *i_name->asString();
   internal_capitalize_name( name );

   // should we delete this string?
   if( i_value == 0 || i_value->isNil() )
   {
      r->removeHeader( *i_name->asString() );
   }
   else if ( i_value->isString() )
   {
      r->setHeader( name, *i_value->asString() );
   }
   else
   {
      r->setHeader( name, i_value->describe() );
   }

   ctx->returnFrame();
}


/*#
   @method getHeader Reply
   @brief Retrieves the value of a given header.
   @param name The name of the header to be queried.
   @return If the header is set, its value as a string; false otherwise.

   The header @b name is automatically capitalized at the beginning and after each '-'
   symbol.

   @note Pay attention to the fact that the maps in which headers are stored are
   case-sensitive.
*/

FALCON_DECLARE_FUNCTION(getHeader, "name:S,value:[S]")
FALCON_DEFINE_FUNCTION_P1(getHeader)
{
   Reply* r = ctx->tself<Reply*>();

   Item* i_name = ctx->param(0);

   if( i_name == 0 || ! i_name->isString() )
   {
      throw paramError();
   }

   String name = *i_name->asString();
   if ( name.size() == 0 )
   {
      return;
   }

   internal_capitalize_name( name );

   String value;
   if( r->getHeader( name, value ) )
   {
      ctx->returnFrame(FALCON_GC_HANDLE( new String(value) ) );
   }
   else
   {
      ctx->returnFrame();
   }
}


/*#
   @method getHeaders Reply
   @brief returns the list of headers that will be sent (or have been sent).
   @return A dictionary of strings containing all the headers prepared for being sent.

   This method returns a map containing all the headers that the WOPI system is going
   to send, or that have been sent if output is already started.

   The map is a snapshot of the current status of the Reply. Headers can be manipulated
   only through @a Reply.setHeader. Any change to this structure won't be reflected
   on the actual reply headers, and similarly, any change in the Reply headers will
   not be reflected into this value.
*/

FALCON_DECLARE_FUNCTION(getHeaders, "")
FALCON_DEFINE_FUNCTION_P1(getHeaders)
{
   Reply* r = ctx->tself<Reply*>();
   ItemDict* id = r->getHeaders();
   ctx->returnFrame( FALCON_GC_HANDLE(id) );
}

/*#
   @method ctype Reply
   @brief Helper creating a Content-Type header.
   @param type Main MIME type of the data that shall be sent to output.
   @optparam subtype MIME Subtype that shall be sent to the output.
   @optparam charset MIME Charset encoding of the output (if text).

   Creates a Content-Type: &lt;type&gt;/&lt;subtype&gt;; charset=&lt;charset&gt;
   field in the headers. It's just a shortcut.

   @note Many functions in the module suppose that the output will be utf-8.
*/

FALCON_DECLARE_FUNCTION(ctype, "type:S,subtype:[S],charset:[S]")
FALCON_DEFINE_FUNCTION_P1(ctype)

{
   Reply* r = ctx->tself<Reply*>();

   Item* i_type = ctx->param(0);
   Item* i_subtype = ctx->param(1);
   Item* i_charset = ctx->param(2);

   if( (i_type == 0 || ! i_type->isString() )
       || ( i_subtype != 0 && !(i_subtype->isString() || i_subtype->isNil() ))
       || ( i_charset != 0 && !(i_charset->isString() || i_charset->isNil() ))
   )
   {
      throw paramError();
   }

   if( i_charset == 0 || i_charset->isNil() )
   {
      if( i_subtype == 0 || i_subtype->isNil() )
         r->setContentType( *i_type->asString() );
      else
         r->setContentType( *i_type->asString(),*i_subtype->asString() );
   }
   else
   {
      r->setContentType( *i_type->asString(),
            i_subtype!= 0 && i_subtype->isString() ? *i_subtype->asString():"",
            *i_charset->asString() );
   }

   ctx->returnFrame();
}

}


//============================================================================================
// Reply class
//

ClassReply::ClassReply():
         Class("%Reply")
{
   addProperty("status", &get_status, &set_status);
   addProperty("reason", &get_reason, &set_reason);
   addProperty("isSent", &get_isSent);

   addMethod( new Function_commit );
   addMethod( new Function_ctype );
   addMethod( new Function_getHeader );
   addMethod( new Function_setHeader );
   addMethod( new Function_getHeaders );
   addMethod( new Function_setCookie );
   addMethod( new Function_clearCookie );
   addMethod( new Function_redirect );
}


ClassReply::~ClassReply()
{
}


void ClassReply::dispose( void* ) const
{
   // do nothing
}


void* ClassReply::clone( void* ) const
{
   // uncloneable
   return 0;
}


void* ClassReply::createInstance() const
{
   // abstract class
   return 0;
}

void ClassReply::gcMarkInstance( void* instance, uint32 mark ) const
{
   Reply* r = static_cast<Reply*>(instance);
   r->gcMark(mark);
}

bool ClassReply::gcCheckInstance( void* instance, uint32 mark ) const
{
   Reply* r = static_cast<Reply*>(instance);
   return r->module()->currentMark() >= mark;
}

}
}

/* end of classreply.cpp */
