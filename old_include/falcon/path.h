/*
   FALCON - The Falcon Programming Language.
   FILE: path.h

   RFC 3986 compliant file path definition.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Feb 2008 22:03:00 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_PATH_H
#define FALCON_PATH_H

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/gcalloc.h>

namespace Falcon
{
class URI;

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

class FALCON_DYN_CLASS Path: public GCAlloc
{
   String m_path;

   String m_device;
   String m_location;
   String m_file;
   String m_extension;

   bool m_bValid;

   // resStart is always 1

   URI* m_owner;

   void compose();
public:

   /** Empty constructor. */
   Path();

   /** Path-in-uri constructor */
   Path( URI *owner );

   /** Path constructor from strings. */
   Path( const String &path ):
      m_bValid( true ),
      m_owner(0)
   {
      set( path );
   }

   /** Copy constructor.
      Copies the other path as-is.
   */
   Path( const Path &other ):
      m_bValid( true ),
      m_owner(0)
   {
      copy( other );
   }

   /** Copy another path.
      Copies the other path as-is.
   */
   void copy( const Path &other );

   /** Set a path from RFC 3986 format. */
   bool set( const String &p );

   /** Retrurn the path in RFC 3986 format. */
   const String &get() const { return m_path; }

   /** Returns a path in MS-Windows format. */
   String getWinFormat() const { String fmt; getWinFormat( fmt ); return fmt; }

   /** Stores this path in windows format in a given string. */
   void getWinFormat( String &str ) const;

   /** Get the resource part (usually the disk specificator). */
   String getResource() const { String fmt; getResource( fmt ); return fmt; }

   /** Stores the resource part in a given string.
      If the path has not a resource part, the string is also cleaned.
      \param str the string where to store the resource part.
      \return true if the path has a resource part.
   */
   bool getResource( String &str ) const;

   /** Get the location part (path to file) in RFC3986 format.
   */
   String getLocation() const { String fmt; getLocation( fmt ); return fmt; }

   /** Gets the location part, eventually including the resource specificator if present. */
   bool getFullLocation( String &str ) const;
   
   /** Gets the location part, eventually including the resource specificator if present. */
   String getFullLocation() const { String fmt; getFullLocation( fmt ); return fmt; }

   /** Stores the resource part in a given string.
      If the path has not a location part, the string is also cleaned.
      \param str the string where to store the location part.
      \return true if the path has a location part.
   */
   bool getLocation( String &str ) const;

   /** Get the location part (path to file) in MS-Windows format. */
   String getWinLocation() const { String fmt; getWinLocation( fmt ); return fmt; }

   /** Stores the location part in a given string in MS-Windows format.
      If the path has not a location part, the string is also cleaned.
      \param str the string where to store the location part.
      \return true if the path has a location part.
   */
   bool getWinLocation( String &str ) const;

   /** Get Windows disk specificator + location (windows absolute path).

      \param str the string where to store the location part.
      \return true if the path has a disk or location part.
   */
   bool getFullWinLocation( String &str ) const;

   /** returns the disk specificator + location (windows absolute path) */
   String getFullWinLocation() const { String fmt; getFullWinLocation( fmt ); return fmt; } 

   /** Get the filename part.
      This returns the file and extension parts separated by a '.'
   */
   String getFilename() const { String fmt; getFilename( fmt ); return fmt; }

   /** Stores the filename part in a given string.
      If the path has not a filename part, the string is also cleaned.
      \param str the string where to store the filename part.
      \return true if the path has a filename part.
   */
   bool getFilename( String &str ) const;

   /** Get the file part alone (without extension). */
   String getFile() const { String fmt; getFile( fmt ); return fmt; }

   /** Get the file part alone (without extension).
      If the path has not a filename part, the string is also cleaned.
      \param str the string where to store the filename part.
      \return true if the path has a filename part.
   */
   bool getFile( String &str ) const;


   /** Get the extension part. */
   String getExtension() const { String fmt; getExtension( fmt ); return fmt; }

   /** Stores the extension part in a given string.
      If the path has not a extension part, the string is also cleaned.
      \param str the string where to store the extension part.
      \return true if the path has a extension part.
   */
   bool getExtension( String &str ) const;

   /** Sets the resource part. */
   void setResource( const String &res );

   /** Sets the location part in RFC3986 format. */
   void setLocation( const String &loc );

   /** Sets both the resource and the location in one step. 
      If the parameter is empty, both location and resource are cleared.
      
      If the parameter is just a resource specificator (i.e. "C:" or "/C:"),
      then the location is cleared.
      
      If it's just a location, then the resource is cleared.

      May return false if the parsing of the new content fails.
      \param floc Full location.
      \return true if correctly parsed.
   */
   bool setFullLocation( const String &floc );

   /** Sets the file part. */
   void setFile( const String &file );

   /** Sets the filename part (both file and extension). */
   void setFilename( const String &fname );

   /** Sets the extension part. */
   void setExtension( const String &extension );

   /** Returns true if this path is an absolute path. */
   bool isAbsolute() const;

   /** Returns true if this path defines a location without a file */
   bool isLocation() const;

   /** Returns true if the path is valid.
      Notice that an empty path is still valid.
   */
   bool isValid() const { return m_bValid; }

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
   void join( const String &loc, const String &name, const String &ext );

   /** Joins a path divided into its constituents into this path.
      \param res the resource locator (i.e. disk unit)
      \param loc the path location of the file.
      \param name the filename.
      \param ext the file extension.
      \param bWin true if the location may be in MS-Windows format (backslashes).
   */
   void join( const String &res, const String &loc, const String &name, const String &ext );

   /** Add a path element at the end of a path.
      This extens the path adding some path element after the currently
      existing location portion. Leading "/" in npath, or trailing "/" in this
      path are ignored, and a traling "/" is forcefully added if there is a file
      element. In example, adding p1/p2 or /p1/p2 through this method:

      /C:file.txt  => /C:/p1/p2/file.txt
      /path/ => /path/p1/p2
      /path/file.txt => /path/p1/p2/file.txt
   */
   void extendLocation( const String &npath );

   Path & operator =( const Path &other ) { copy( other ); return *this; }
   bool operator ==( const Path &other ) const { return other.m_path == m_path; }
   bool operator !=( const Path &other ) const { return other.m_path != m_path; }
   bool operator <( const Path &other ) const { return m_path < other.m_path; }


   /** Converts an arbitrary MS-Windows Path to URI path.
    An URI valid path starts either with a filename or with a "/"
    if absolute. It can't start with a MS-Windows disk or unit
    specifier as c:\\. Also, it can't contain backslashes.

    This static function transforms the path parameter in place
    so that it is changed into a valid URI path. For example:
    \code
       path\\to\\file.txt  => path/to/file.txt
       \\path\\to\\file.txt => /path/to/file.txt
       c:\\path\\to\\file.txt => /c:/path/to/file.txt
    \endcode
    @param path the path to be converted on place
   */
   static void winToUri( String &path );

   /** Converts an arbitrary URI path into a MS-Windows path.
    An URI valid path starts either with a filename or with a "/"
    if absolute. It can't start with a MS-Windows disk or unit
    specifier as c:\\. Also, it can't contain backslashes.

    This static function transforms the path parameter in place
    so that it is changed into a valid MSWindows path. For example:
    \code
        path/to/file.txt => path\\to\\file.txt
        /path/to/file.txt => \\path\\to\\file.txt
        /c:/path/to/file.txt = c:\\path\\to\\file.txt
    \endcode

      @note This function won't complain nor emit any warning
      if path is not a valid URI path in the first place.
      It's pretty dumb, but efficient.

    @param path the path to be converted on place
    @see getWinFormat
   */
   static void uriToWin( String &path );
};

}

#endif
