/*
   FALCON - The Falcon Programming Language.
   FILE: uri.cpp

   Falcon core module -- Interface to URI class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 21:49:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/uri.cpp"

#include <falcon/classes/classuri.h>
#include <falcon/itemid.h>

#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/uri.h>
#include <falcon/path.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/function.h>

#include <falcon/stderrors.h>

namespace Falcon {

/*# @class URI
 @brief Generic URL parser entity.
 @param uri A string representing an URI.

 This class parses and then holds a parsed Universal Resource Identifier
 as per RFC3986. Most of the functions dealing with abstract I/O
 accept this class instances as a valid parameters, and the class has
 methods and property accessors to synthetize valid URI representation
 out of its components.

   @prop scheme Gets or sets the URI scheme part (without the :// separator).
   @prop path Gets or sets the path part (as "/a/b/c/file.ext").
   @prop query Gets or sets the query part ("var=value&var=value..." that goes after "?").
   @prop host Gets or sets the host from the auth element.
   @prop port Gets or sets the port from the auth element.
   @prop user Gets or sets the user from the auth element.
   @prop pwd Gets or sets the password from the auth element.
   @prop fragment Gets or sets the fragment part (the element going after "#").

 @note This class has type dignity and a numeric type id.
*/

//====================================================
// Methods
//

/*# @method setq URI
 @brief Sets a query element
 @param key The key in the query part to be set
 @param value The value in the query part to be set

 This method sets a query variable @b key to the given @b value.
 If the parameters are not string, they are converted to strings prior
 being set in the query part.

 Necessary escaping is perfomred by the method.
 */
class MethodSetq: public Function
{
public:
   MethodSetq():
      Function( "setq" )
   {
      signature("X,X");
      addParam("key");
      addParam("value");
   }         
   virtual ~MethodSetq() {}
   
   virtual void invoke( VMContext* ctx, int32 pCount = 0 );
};

/*# @method setq URI
 @brief Retrieves a query element
 @param key The key in the query part to be retrieved
 @optparam dflt An optional default value to be returned if the key is not found
 @raise AccessError if the @b key is not set and @b dflt is not given.

 Necessary escaping is perfomred by the method.
 */
class MethodGetq: public Function
{
public:
   MethodGetq( ):
      Function( "getq" )
   {
      signature("X,[X]");
      addParam("key");
      addParam("default");
   }         
   virtual ~MethodGetq() {}
   
   virtual void invoke( VMContext* ctx, int32 pCount = 0 );
};

 
//=========================================================
// Properties
//

/*# @property encoded URI
 @brief Gets or sets the full URI representation.
 @raise ParseError if setting a malformed URI.

 Accessing this property returns the full URI as synthesized using
 the various components that are currently set.

 Setting this property changes the whole contents of the URI, parsing
 the source string as a new full URI.

 In case the URI is malformed, a ParseError is raised.

*/
static void set_encoded( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   URI* uri = static_cast<URI*>(instance);
   if( ! uri->parse( *value.asString() ) )
   {
      throw new ParseError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }   
}


static void get_encoded( const Class*, const String&, void* instance, Item& value )
{
   URI* uric = static_cast<URI*>(instance);
   value = FALCON_GC_HANDLE( new String( uric->encode() ) );
}


static void set_scheme( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   URI* uric = static_cast<URI*>(instance);
   uric->scheme(*value.asString());
}


static void get_scheme( const Class*, const String&, void* instance, Item& value )
{
   URI* uric = static_cast<URI*>(instance);
   value = FALCON_GC_HANDLE( new String( uric->scheme() ) );
}

/*#
 @property auth URI
 @brief Gets or sets the authorization part as a whole.

 The authorization part is the combination of user, password, host and port components.

 The authorization part is formatted as here indicated:
 @code
    user:password@host:port
 @endcode

 Each component is optional. If some parts only are present,the authorization is
 represented as

 @code
    host             // host only
    :port            // port only
    host:port        // host and port
    user@            // user only
    user@host        // host and user
    :pwd@            // password only
    user@:port       // user and port
    :pwd@:port       // password and port
 @endcode

 and so on.

 It is valid to set the auth element to an empty string to clear all its components.
 */

static void set_auth( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   URI* uric = static_cast<URI*>(instance);
   if( ! uric->auth().parse( *value.asString() ) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}


static void get_auth( const Class*, const String&, void* instance, Item& value )
{
   URI* uric = static_cast<URI*>(instance);
   value = FALCON_GC_HANDLE( new String( uric->auth().encode() ));
}



static void set_path( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   URI* uric = static_cast<URI*>(instance);
   if( ! uric->path().parse( *value.asString() ) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}


static void get_path( const Class*, const String&, void* instance, Item& value )
{
   URI* uric = static_cast<URI*>(instance);
   value = FALCON_GC_HANDLE( new String( uric->path().encode() ));
}


static void set_query( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   URI* uric = static_cast<URI*>(instance);
   if( ! uric->query().parse( *value.asString() ) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }   
}


static void get_query( const Class*, const String&, void* instance, Item& value )
{
   URI* uric = static_cast<URI*>(instance);
   value = FALCON_GC_HANDLE( new String( uric->query().encode() ));
}


static void set_host( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   URI* uric = static_cast<URI*>(instance);
      
   uric->auth().host( *value.asString() );
}

static void get_host( const Class*, const String&, void* instance, Item& value )
{
   URI* uric = static_cast<URI*>(instance);
   value = FALCON_GC_HANDLE( new String( uric->auth().host() ));
}


static void set_fragment( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   URI* uric = static_cast<URI*>(instance);
   uric->fragment( *value.asString() );
}


static void get_fragment( const Class*, const String&, void* instance, Item& value )
{
   URI* uric = static_cast<URI*>(instance);
   value = FALCON_GC_HANDLE( new String( uric->fragment() ));
}



static void set_port( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   URI* uric = static_cast<URI*>(instance);
   uric->auth().port( *value.asString() );
}


static void get_port( const Class*, const String&, void* instance, Item& value )
{
   URI* uric = static_cast<URI*>(instance);
   value = FALCON_GC_HANDLE( new String( uric->auth().port() ));
}

   
static void set_user( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   URI* uric = static_cast<URI*>(instance);
   uric->auth().user( *value.asString() );
}


static void get_user( const Class*, const String&, void* instance, Item& value )
{
   URI* uric = static_cast<URI*>(instance);
   value = FALCON_GC_HANDLE( new String( uric->auth().user() ));
}


static void set_pwd( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   URI* uric = static_cast<URI*>(instance);
   uric->auth().password( *value.asString() );
}


static void get_pwd( const Class*, const String&, void* instance, Item& value )
{
   URI* uric = static_cast<URI*>(instance);
   value = FALCON_GC_HANDLE( new String( uric->auth().password() ));
}   
 

//==================================================================
// Main class
//==================================================================

ClassURI::ClassURI():
   Class("URI", FLC_CLASS_ID_URI)
{   
   addProperty( "encoded", &get_encoded, &set_encoded );
   addProperty( "scheme", &get_scheme, &set_scheme );
   addProperty( "auth", &get_auth, &set_auth );
   addProperty( "path", &get_path, &set_path );
   addProperty( "query", &get_query, &set_query );
   addProperty( "fragment", &get_fragment, &set_fragment );
   addProperty( "host", &get_host, &set_host );
   addProperty( "port", &get_port, &set_port );
   addProperty( "user", &get_user, &set_user );
   addProperty( "pwd", &get_pwd, &set_pwd );  
   
   addMethod( new MethodGetq );
   addMethod( new MethodSetq );
}

ClassURI::~ClassURI()
{}

 
void ClassURI::dispose( void* instance ) const
{
   URI* uc = static_cast<URI*>(instance);
   delete uc;
}

void* ClassURI::clone( void* instance ) const
{
   URI* uc = static_cast<URI*>(instance);
   return new URI(*uc);
}

void ClassURI::gcMarkInstance( void* instance, uint32 mark ) const
{
   URI* uc = static_cast<URI*>(instance);
   uc->gcMark(mark);
}

bool ClassURI::gcCheckInstance( void* instance, uint32 mark ) const
{
   URI* uc = static_cast<URI*>(instance);
   return uc->currentMark() >= mark;
}


void ClassURI::store( VMContext*, DataWriter* stream, void* instance ) const
{
   URI* uc = static_cast<URI*>(instance);
   stream->write(uc->encode());
}


void ClassURI::restore( VMContext* ctx, DataReader* stream ) const
{
   String uriName;
   stream->read( uriName );
   URI* uc = new URI();
   try {
      uc->parse( uriName );
      ctx->pushData( Item( this, uc ) );
   }
   catch( ... ) {
      delete uc;
      throw;
   }
}
   
void* ClassURI::createInstance() const
{
   return new URI();
}


bool ClassURI::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   URI* uc = static_cast<URI*>(instance );
   
   if ( pcount >= 1 )
   {
      Item& other = *ctx->opcodeParams(pcount);
      if( other.isString() )
      {
         if( ! uc->parse( *other.asString() ) )
         {
            delete uc;
            throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC )
               .origin(ErrorParam::e_orig_mod) );
         }
         return false;
      }

      if( other.asClass()->isDerivedFrom(this) )
      {
         void* inst = other.asClass()->getParentData(this, other.asInst());
         if( inst != 0 )
         {
            const URI* orig = static_cast<URI*>(inst);
            uc->copy( *orig );
         }
         return false;
      }

      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
               .extra( "S|URI" )
               .origin(ErrorParam::e_orig_mod) );
   }

   return false;
}

void ClassURI::op_toString( VMContext* ctx, void* self ) const
{
   URI* uc = static_cast<URI*>(self);
   ctx->topData() = FALCON_GC_HANDLE(new String(uc->encode()));
}

//=========================================================
// Methods
//

void MethodSetq::invoke( VMContext* ctx, int32 )
{
   Item* iKey = ctx->param(0);
   Item* iValue = ctx->param(1);
   
   if( iKey == 0 || ! iKey->isString() || 
       iValue == 0 || ! ( iValue->isString() || iValue->isNil() ) )
   {
      throw paramError( __LINE__, SRC );
   }
   
   URI* uc = static_cast<URI*>(ctx->self().asInst());
   if( iValue->isString() )
   {
      uc->query().put( *iKey->asString(), *iValue->asString() );
   }
   else
   {
      uc->query().remove( *iKey->asString() );
   }
   
   ctx->returnFrame();
}
   
      
void MethodGetq::invoke( VMContext* ctx, int32 )  
{
   Item* iKey = ctx->param(0);
   Item* iValue = ctx->param(1);
   
   if( iKey == 0 || ! iKey->isString() )
   {
      throw paramError( __LINE__, SRC );
   }
   
   URI* uc = static_cast<URI*>(ctx->self().asInst());
   
   String value;
   if( uc->query().get( *iKey->asString(), value ) )
   {
      ctx->returnFrame( value );
   }
   else
   {
      if( iValue != 0 )
      {
         ctx->returnFrame( *iValue );
      }
      else
      {
         ctx->returnFrame();
      }
   }
}

}

/* end of uri.cpp */
