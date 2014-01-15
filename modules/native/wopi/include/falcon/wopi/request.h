/*
   FALCON - The Falcon Programming Language.
   FILE: request.h

   Web Oriented Programming Interface

   Object encapsulating requests.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 13 Feb 2010 12:29:23 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_WOPI_REQUEST
#define FALCON_WOPI_REQUEST

#include <falcon/wopi/utils.h>
#include <falcon/wopi/parthandler.h>

#include <falcon/uri.h>
#include <falcon/stream.h>

namespace Falcon {
namespace WOPI {

class ModuleWopi;

class Request
{
public:

   Request( ModuleWopi* host = 0 );
   virtual ~Request();

   //=========================================================
   // Main operations
   //

   //! Do a complete parse of the whole input (headers and body)
   virtual void parse( Stream* input );

   //! parse the header part.
   /** \note this inserts Falcon GC relevant objects in the GC,
    *  if invoked from outside the VM, wrap in gc-disabled zone.
    *  (Also, the Request object should be already locked/reachable from GC)
    */
   virtual void parseHeader( Stream* input );

   //! Parses the body.
   /** \note this inserts Falcon GC relevant objects in the GC,
    *  if invoked from outside the VM, wrap in gc-disabled zone.
    *  (Also, the Request object should be already locked/reachable from GC)
    */
   virtual void parseBody( Stream* input );

   //! Reads relevant CGI-environ variables
   virtual void parseEnviron();

   //=========================================================

   bool getField( const String& fname, String& value ) const;
   bool getField( const String& fname, Item& value ) const;

   bool getFieldOrArray( const String& fname, Item& value ) const
   {
      if ( ! getField( fname, value ) )
         return getField(fname + "[]", value );
      return true;
   }

   void fwdGet( String& fwd, bool all=false ) const;
   void fwdPost( String& fwd, bool all=false ) const;

   bool setURI( const String& uri );
   const URI& parsedUri() const { return m_uri; }
   URI& parsedUri() { return m_uri; }
   const String& getUri() const { return m_sUri; }

   bool addPartData( byte* data, int length );

   bool parsePartHeaderLine( const String& line );
   const PartHandler& partHandler() const { return m_MainPart; }
   PartHandler& partHandler() { return m_MainPart; }

   void startedAt( Falcon::numeric t ) { m_startedAt = t; }
   Falcon::numeric startedAt() const { return m_startedAt; }

   ItemDict* gets() const { return m_gets; }
   ItemDict* posts() const { return m_posts; }
   ItemDict* cookies() const { return m_cookies; }
   ItemDict* headers() const { return m_headers; }

   void gcMark( uint32 m );
   inline uint32 currentMark() const { return m_mark; }


   /** Adds uploaded parts and process multipart fields if the request has a multipart body.

       This should ALWAYS be called after the process is fully parsed to
       translate each parts into script-available objects.

       \return true If the request has a multipart body.
   */
   virtual bool processMultiPartBody();

   ModuleWopi* module() const { return m_module; }
   void module( ModuleWopi* mod ) { m_module = mod; }

   // Generic request informations
   String m_protocol;
   String m_method;

   String m_location;
   String m_filename;
   String m_path_info;
   String m_args;
   String m_remote_ip;

   // Authorization fields
   String m_ap_auth_type;
   String m_user;

   // Post-request parse fields.
   String m_content_type;
   String m_content_encoding;

   PartHandler m_MainPart;

   // quantitative informations
   int64 m_content_length;
   int64 m_bytes_received;
   int64 m_request_time;
   numeric m_startedAt;
   String m_sUri;
   URI m_uri;

protected:
   void forward( const ItemDict& main, const ItemDict& aux, String& fwd, bool all ) const;

   //! Creates an uploaded element (in the post fields) out of the data in partHandler
   void addUploaded( PartHandler* ph, const String& prefix = "" );

   // host module
   ModuleWopi* m_module;

   // dictionaries
   ItemDict* m_gets;
   ItemDict* m_posts;
   ItemDict* m_cookies;
   ItemDict* m_headers;

   uint32 m_mark;

   bool m_bPostInit;

private:
   static void handleEnvStr( const Falcon::String& key, const Falcon::String& value, void *data );
   void addHeaderFromEnv( const Falcon::String& key, const Falcon::String& value );
};

}
}

#endif

/* end of request.h */
