/*
   FALCON - The Falcon Programming Language.
   FILE: uri.h

   RFC 3986 - Uniform Resource Identifier
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Feb 2008 12:23:28 +0100
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#ifndef FALCON_URI_H
#define FALCON_URI_H

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/userdata.h>
#include <falcon/genericmap.h>

namespace Falcon
{

/** Falcon path representation.

   This class is actually a string wrapper which parses the path and builds it as necessary.

   With respect to a string, 0 overhead is required.

   However, notice that this class cannot be used as a garbage string, and must be wrapped
   into a UserData to be part of a falcon object.

   Path must be provided in Falcon format (RFC 3986): path elements must be separated by forward
   slashes and resource identifiers must be preceded by a single "/"; in example:
   \code
      /C:/falcon/file.fal
   \endcode
   With a resource identifier, the first "/" is optional when setting the path, but the
   internal representation will be normalized so that it is present.

   Methods to transfrorm this representation to and from MS-Windows path are provided.

   The path is not internally checked, by this class, so any string may be set,
   but it may get checked i.e. when insernted in a URI.
*/

class Path: public BaseAlloc
{
   String m_path;

   // resStart is always 1
   uint32 m_resEnd;
   uint32 m_pathStart;
   uint32 m_pathEnd;
   uint32 m_fileStart;
   uint32 m_fileEnd;
   uint32 m_extStart;

   /** Analyze the path, splitting its constituents.
      \param isWin true to perform also \\ -> / conversion while parsing.
   */
   void analyze( bool isWin );

public:

   /** Empty constructor. */
   Path();

   /** Path constructor from strings. */
   Path( const String &path )
   {
      set( path );
   }

   /** Path constructor from strings.
      This constiuctor allows to select between MS-Windows path format or Falcon path format.
   */
   Path( const String &path, bool winFormat )
   {
      if ( winFormat )
         setFromWinFormat( path );
      else
         set( &path );
   }

   /** Copy constructor.
      Copies the other path as-is.
   */
   Path( const Path &other );


   /** Set a path from RFC 3986 format. */
   void set( const String &p );

   /** Set a path having MS-Windows format */
   void setFromWinFormat( const String &p );

   /** Retrurn the path in RFC 3986 format. */
   const String &get() const { return m_path; }

   /** Returns a path in MS-Windows format. */
   String &getWinFormat() const { String fmt; getWinFormat( fmt ); return fmt; }

   /** Stores this path in windows format in a given string. */
   void getWinFormat( String &str ) const;

   /** Get the resource part (usually the disk specificator). */
   String &getResource() const { String fmt; getResource( fmt ); return fmt; }

   /** Stores the resource part in a given string. */
   void getResource( String &str ) const;

   /** Get the location part (path to file) in RFC3986 format. */
   String &getLocation() const { String fmt; getLocation( fmt ); return fmt; }

   /** Stores the resource part in a given string. */
   void getLocation( String &str ) const;

   /** Get the location part (path to file) in MS-Windows format. */
   String &getWinLocation() const { String fmt; getWinLocation( fmt ); return fmt; }

   /** Stores the location part in a given string in MS-Windows format. */
   void getWinLocation( String &str ) const;

   /** Get the filename part. */
   String &getFilename() const { String fmt; getFilename( fmt ); return fmt; }

   /** Stores the filename part in a given string. */
   void getFilename( String &str ) const;

   /** Get the extension part. */
   String &getExtension() const { String fmt; getExtension( fmt ); return fmt; }

   /** Stores the extension part in a given string. */
   void getExtension( String &str ) const;

   /** Sets the resource part. */
   void setResource( const String &res );

   /** Sets the location part in RFC3986 format. */
   void setLocation( const String &loc );

   /** Sets the location part in MS-Windows format. */
   void setWinLocation( const String &loc );

   /** Sets the file part. */
   void setFile( const String &file );

   /** Sets the extension part. */
   void setExtension( const String &extension );

   /** Returns true if this path is an absolute path. */
   bool isAbsolute() const;

   /** Returns true if this path defines a location without a file */
   bool isLocation() const;

   /** Splits the path into its constituents.
      This version would eventually put the resource part in the first parameter.
      \param loc a string where the location will be placed.
      \param name a string where the filename in this path will be placed.
      \param ext a string where the file extension will be placed.
   */
   void split( String &loc, String &name, String &ext );

   /** Splits the path into its constituents.
      \param res a string where the resource locator will be placed.
      \param loc a string where the location will be placed.
      \param name a string where the filename in this path will be placed.
      \param ext a string where the file extension will be placed.
   */
   void split( String &res, String &loc, String &name, String &ext );

   /** Splits the path into its constituents.
      This version will convert the output loc parameter in MS-Windows path format
         (backslashes).
      \param res a string where the resource locator will be placed.
      \param loc a string where the location will be placed.
      \param name a string where the filename in this path will be placed.
      \param ext a string where the file extension will be placed.
   */
   void splitWinFormat( String &res, String &loc, String &name, String &ext );

   /** Joins a path divided into its constituents into this path.
      Using this version it is not possible to set a resource locator (i.e. a disk unit).

      \param loc the path location of the file.
      \param name the filename.
      \param ext the file extension.
   */
   void join( String &loc, String &name, String &ext );

   /** Joins a path divided into its constituents into this path.
      \param res the resource locator (i.e. disk unit)
      \param loc the path location of the file.
      \param name the filename.
      \param ext the file extension.
   */
   void join( String &res, String &loc, String &name, String &ext );

   /** Joins a path divided into its constituents into this path.
      \param res the resource locator (i.e. disk unit)
      \param loc the path location of the file in MS-Windows format.
      \param name the filename.
      \param ext the file extension.
   */
   void joinWinFormat( String &res, String &loc, String &name, String &ext );
};


/** RFC 3986 - Uniform Resource Identifier.

   This class offer falcon engine and its users an interface to URI.
*/
FALCON_DYN_CLASS class URI: public BaseAlloc
{
   /** A copy of the original string, for diagniostics.*/
   String m_original;

   /** The final normalized and encoded URI. */
   String m_encoded;

   /** URI scheme (e.g. http) */
   String m_scheme;

   /** False if this URI is not valid. */
   bool m_bValid;

   // Authority ----------------------
   /** User or user:password. */
   String m_userInfo;

   /** Host part of the URI. */
   String m_host;

   /** Port part of the URI. */
   String m_port;

   /** Path.
      Virtually divided into 4 elements:
      \code
      /[resource:]path/file.ext
      \endcode

      but calculus on path is done realtime.

      \note relative paths (not starting with /) cannot have a resource part.
   */
   Path m_path;

   /** Query part.
      Map of string->string
   */
   Map m_query;

   /** Fragment. */
   String m_fragment;

   /** Parse a given string in the final elements of the uri. */
   bool parse( const String &u );

   /** Encodes a prebuilt string which is then placed in the m_encoded field. */
   void encode( const String &u );


public:

   /** Empty constructor.
      Creates an empty URI. To be filled at a later time.
   */
   URI();

   /** Complete URI constructor.
      Decodes the uri and parses it in its fields.

      In case URI is not valid, isValid() will return false after construction.
   */
   URI( const String &suri );

   /** Copy constructor.
      Copies everything in the other URI, including validity.
   */
   URI( const URI &other );

   /** Parses the given string into this URI.
      \param newUri the new URI to be parsed.
      \return true on success, false if the given string is not a valid URI.
   */
   bool uri( const String &newUri );

   /** Returns the current URI.
      This method eventually builds a new URI from the internally parsed data
      and returns it.
   */
   const String &uri();

   //  A bit of parsing support.
   /** Character is a reserved delimiter under RFC3986 */
   static bool isResDelim( uint32 chr );

   /** Character is a general delimiter under RFC3986 */
   static bool isGenDelim( uint32 chr );

   /** Character is a subdelimiter under RFC4986 */
   static bool isSubDelim( uint32 chr );

   /** Normalzies the URI sequence.
      Transforms ranges of ALPHA
      (%41-%5A and %61-%7A), DIGIT (%30-%39), hyphen (%2D), period (%2E),
      underscore (%5F), or tilde (%7E) into their coresponding values.

      Other than that, it transforms %20 in spaces.

      The normalization is performed on a result string.

      Every string set in input by any method in this class is normalized
      prior to storage.

      However, notice that parts in the class may be invalid URI elements
      if extracted as is, as they are stored as Falcon international strings.
      Encoding to URI conformance happens only when required.

      \param part the element of the URI (or the complete URI) to be normalized.
      \param result the string where the normalization is performed.
   */
   static void normalize( const String &part, String &result );

   /** Returns current scheme. */
   const String &scheme() const { return m_scheme; }

   /** Sets a different scheme for this URI.
      This will invalidate current URI until next uri() is called.
   */
   void scheme( const String &s );

   /** Returns current userInfo. */
   const String &userInfo() const { return m_userInfo; }

   /** Sets a different userInfo for this URI.
      This will invalidate current URI until next uri() is called.
   */
   void userInfo( const String &s );

   /** Returns current host. */
   const String &host() const { return m_host; }

   /** Sets a different host for this URI.
      This will invalidate current URI until next uri() is called.
   */
   void host( const String &h );

   /** Returns current port. */
   const String &port() const { return m_port; }

   /** Sets a different port for this URI.
      This will invalidate current URI until next uri() is called.
   */
   void port( const String &h );

   /** Returns current path. */
   const String &path() const { return m_path.path(); }

   /** Returns current path as a path class. */
   const Path &pathElement() const { return m_path; }

   /** Sets a different path for this URI.
      This will invalidate current URI until next uri() is called.
   */
   void path( const String &p );

   /** Sets a different path for this URI.
      This will invalidate current URI until next uri() is called.
   */
   void path( const Path &p );

   /** Returns true if the query part has this field. */
   bool hasField( const String &f ) const ;

   /** Returns the required value, if it exists. */
   bool getField( const String &key, String &value ) const;

   /** Sets a given query field.
      As a convention, if the string contains only a single 0 character (NUL), the
      final result won't include = in the query part, while an empty string will
      result in a query string containing only a keu and a "=" followed by nothing.
      Strings longer than 1 element are not interpreted this way, so a nul followed
      by some data would be rendere as "%00" followed by the rest of the value.
   */
   void setField( const String &key, const String &value );

   /** Removes a query field. */
   bool removeField( const String &key );

   /** Returns the fragment part. */
   const String &fragment() const { return m_fragment; }

   /** Sets a different fragment for this URI.
      This will invalidate current URI until next uri() is called.
   */
   void port( const String &h );

   //============================
   // path helpers

   bool isPathAbsolute() const ;

};

//==================================================
// Inline implementation
//

inline static bool URI::isResDelim( uint32 chr )
{
   return isGenDelim( chr ) | isSubDelim( chr );
}

inline static bool URI::isGenDelim( uint32 chr )
{
   return chr == ':' | chr == '/' | chr == '?' |
          chr == '#' | chr == '[' | chr == ']' |
          chr == '@';
}

inline static bool URI::isSubDelim( uint32 chr )
{
   return chr == '!' | chr == '$' | chr == '&' |
          chr == '\'' | chr == '(' | chr == ')' |
          chr == '*' | chr == '+' | chr == ',' |
          chr == ';' | chr == '=';
}

} // Falcon namespace

#endif
