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

#define FALCON_WOPI_MAXMEMUPLOAD_ATTRIB "wopi_maxMemUpload"
#define FALCON_WOPI_TEMPDIR_ATTRIB "wopi_tempDir"

class SessionManager;
class CoreRequest;
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

   /** Create a generic usage temporary file. */
   Stream* makeTempFile( String& fname, int64& le );

   //=========================================================
   // Overridable
   //

   virtual const String& getTempPath() const { return m_sTempPath; }
   virtual int64 getMemoryUpload() const { return m_nMaxMemUpload; }

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

   //! Sets the maximum size that for uploading a part.
   void setMaxMemUpload( int64 mm ) { m_nMaxMemUpload = mm; }

   //! Sets the location for temporary files.
   void setUploadPath( const String& path ) { m_sTempPath = path; }

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

   CoreDict* gets() const { return m_gets; }
   CoreDict* posts() const { return m_posts; }
   CoreDict* cookies() const { return m_cookies; }
   CoreDict* headers() const { return m_headers; }

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

   // Create the headers in a canonical form
   //CoreDict* makeCanonicalHeaders();

   // dictionaries
   CoreDict* m_gets;
   CoreDict* m_posts;
   CoreDict* m_cookies;
   CoreDict* m_headers;

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

   String m_sTempPath;
   int64 m_nMaxMemUpload;

   uint32 m_nSessionToken;

   Falcon::numeric m_startedAt;

   friend class CoreRequest;

   GCLock* m_lockGets;
   GCLock* m_lockPosts;
   GCLock* m_lockHeaders;
   GCLock* m_lockCookies;
};


/** Script side Request wrapper.

    This class implements a Request object as seen by the Falcon script side.

    It's mainly a wrapper for Request class with some utilities for the
    falcon scripts using it.

    A Request can be created elsewhere and passed to this class via
    its init() method, or this class will create its own Request at init().
 */
class CoreRequest: public CoreObject
{
public:

   CoreRequest( const CoreClass* base );
   virtual ~CoreRequest();

   /** Construct this object after the initial creation.

       This allows to build the object after it has been pre-created
       by the virtual machine.

       @param upd The coreclass serving as the generator
    */
   void init( CoreClass* upd, Reply* reply, SessionManager* sm, Request* r=0 );

   Request* base() const { return m_base; }

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

   /** Configure this request using the given module.

      Uses the given module attributes to configure:
      - The temporary upload path (cgi_tempDir).
      - The maximum upload did size (cgi_maxMemUpload).
   */
   void configFromModule( const Module* mod );

   //=====================================================
   // Utilities for script
   //

   /** Get the session manager. */
   SessionManager* smgr() const { return m_sm; }

   //=====================================================
   // Overrides from CoreObject
   //

   virtual CoreObject *clone() const;
   virtual bool setProperty( const String &prop, const Item &value );
   virtual bool getProperty( const String &prop, Item &value ) const;
   virtual void gcMark( uint32 mark );

   static CoreObject* factory( const Falcon::CoreClass* cls, void* ud, bool bDeser );

   Reply* reply() const { return m_reply; }

   bool autoSession() const { return m_bAutoSession; }

   void provider( const String& s ) { m_provider = s; }

protected:
   //! Creates an uploaded element (in the post fields) out of the data in partHandler
   void addUploaded( PartHandler* ph, const String& prefix = "" );

   String m_provider;

   SessionManager* m_sm;
   CoreClass* m_upld_c;
   bool m_bPostInit;

   Request* m_base;
   Reply* m_reply;

   bool m_bAutoSession;
};


}
}

#endif

/* end of request.h */
