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


class SessionManager;
class Reply;

class Request
{
public:

   Request();
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

   /** Create a generic usage temporary file.
    \throw IoError on error.
    * */
   Stream* makeTempFile( String& fname );


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

   /** Adds a temporary file.
      The VM tries to delete all the temporary files during its destructor.
      On failure, it ingores the problem and logs an error in Apache.
   */
   void addTempFile( const Falcon::String &fname );

   /** Gets the list of temporary files.
      Before the VM is destroyed, this should be taken out
      so that it is then possible to get rid of the files.

      Using removeTempFiles after the VM has been destroyed ensures
      that all the streams open by the VM are closed (as this is done
      during the GC step).

      \return an opaque pointer to an internal structure.
   */
   void* getTempFiles() const { return m_tempFiles; }

   /** Removes from the disk a list of temporary files.

      Using removeTempFiles after the VM has been destroyed ensures
      that all the streams open by the VM are closed (as this is done
      during the GC step).

      \param head The valued returned from getTempFiles() before the VM was destroyed.
      \param data Opaque pointer passed as extra data to the error_func (can be 0 if not used).
      \param error_func callback that will be invoked in some file can't be deleted.
   */
   static void removeTempFiles( void* head, void* data, void (*error_func)(const String& msg, void* data) );

   void startedAt( Falcon::numeric t ) { m_startedAt = t; }
   Falcon::numeric startedAt() const { return m_startedAt; }

   ItemDict* gets() const { return m_gets; }
   ItemDict* posts() const { return m_posts; }
   ItemDict* cookies() const { return m_cookies; }
   ItemDict* headers() const { return m_headers; }

   void gcMark( uint32 m );
   inline uint32 currentMark() const { return m_mark; }

   /** Get the session manager. */
   SessionManager* smgr() const { return m_sm; }

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

   inline void provider( const String& s ) { m_provider = s; }

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

   // Create the headers in a canonical form
   //CoreDict* makeCanonicalHeaders();

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

   String m_provider;
   SessionManager* m_sm;
   bool m_bPostInit;
   bool m_bAutoSession;
};

}
}

#endif

/* end of request.h */
