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

#define DEFAULT_SESSION_FIELD "SID"

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

   Request( ModuleWopi* host );
   virtual ~Request();

   //=========================================================
   // Main operations
   //

   //! Do a complete parse of the whole input (headrs and body)
   bool parse( Stream* input );

   //! parse the header part.
   bool parseHeader( Stream* input );

   //! Parses the body.
   bool parseBody( Stream* input );

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

   const String& getSessionFieldName() const { return m_sSessionField; }
   void setSessionFieldName( const String& name ) { m_sSessionField = name; }

   /** Token created by the session manager for this requet.
   */
   uint32 sessionToken() const { return m_nSessionToken; }
   void sessionToken( uint32 st ) { m_nSessionToken = st; }

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

   /** Override this to be called back at first usage.

       Also, set m_bPostInit to true; in this way, this
       virtual function will be called back the first time
       setProperty or getProperty is called on this object.
    */
   virtual void postInit() {}

   /** Adds uploaded parts and process multipart fields if the request has a multipart body.

       This should ALWAYS be called after the process is fully parsed to
       translate each parts into script-available objects.

       \return true If the request has a multipart body.
   */
   virtual bool processMultiPartBody();

   virtual Request *clone() const;

   inline bool autoSession() const { return m_bAutoSession; }

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
   int64 m_request_time;
   int64 m_bytes_sent;
   int64 m_content_length;
   String m_sUri;

protected:
   void forward( const ItemDict& main, const ItemDict& aux, String& fwd, bool all ) const;

   // generate the headers when first requested.
   void makeHeaders();

   //! Creates an uploaded element (in the post fields) out of the data in partHandler
   void addUploaded( PartHandler* ph, const String& prefix = "" );


   // host module
   ModuleWopi* m_module;

   // dictionaries
   ItemDict* m_gets;
   ItemDict* m_posts;
   ItemDict* m_cookies;
   ItemDict* m_headers;

   String m_sSessionField;

   URI m_uri;

   class TempFileEntry
   {
      public:
         Falcon::String m_entry;
         TempFileEntry* m_next;

         TempFileEntry( const Falcon::String &fname ):
            m_entry( fname ),
            m_next(0)
            {}
   };

   // Used to remember which files to delete at end.
   TempFileEntry *m_tempFiles;

   uint32 m_nSessionToken;

   Falcon::numeric m_startedAt;

   uint32 m_mark;

   bool m_bPostInit;
   bool m_bAutoSession;
};

}
}

#endif

/* end of request.h */
