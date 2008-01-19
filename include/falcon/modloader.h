/*
   FALCON - The Falcon Programming Language.
   FILE: flc_modloader.h
   $Id: modloader.h,v 1.9 2007/08/11 18:26:44 jonnymind Exp $

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar ago 3 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#ifndef flc_MODLOADER_H
#define flc_MODLOADER_H

#include <falcon/common.h>
#include <falcon/error.h>
#include <falcon/errhand.h>
#include <falcon/string.h>
#include <falcon/basealloc.h>

namespace Falcon {

class Module;
class Stream;

/** Module Loader support.
   This class enables embedding applications and falcon VM (and thus, Falcon scripts)
   to load modueles.
   \TODO more docs
*/
class FALCON_DYN_CLASS ModuleLoader: public BaseAlloc
{
public:
   /** File type enumeration.
      The module loader tries to determine which type of file
      the it is trying to load (or is told this information
      by the loader user).
   */
   typedef enum
   {
      /** Undetermined / detect. */
      t_none,
      /** The module is a source. */
      t_source,
      /** The module is a falcon FAM module. */
      t_vmmod,
      /** The module is a native .so/.dll. */
      t_binmod,
      /** Special file type.
          Try to determine the type of the file,
          but in case it cannot be determined, defaults to source.
      */
      t_defaultSource
   } t_filetype;

private:
   Module *loadModule_ver_1_0( Stream *in );

protected:
   ErrorHandler *m_errhand;

   Module *loadModule_select_ver( Stream *in );

   /** Discovers the module name given a complete file path.
      \param path the path to a possible falcon module
      \param modNmae the possible falcon module name
   */
   static void getModuleName( const String &path, String &modName );

   /** Tell if the subclass accepts sources or not.
      Vast part of the module loader is configured to behave the same way
      when compilation of source scripts is allowed or not. However, the base
      ModuleLoader class can't compile directly scripts (i.e. can be used to
      run a set of pre-compiled scripts). If a module loader subclass has
      support for sources, this variable should be set to true in its
      constructor.
   */
   bool m_acceptSources;

   /** Path where to search the modules.
      Each entry of the list is one single system path encoded in falcon file name format.
   */
   List m_path;

   /** Basic function used to open modules.
      This virtual function opens a disk file as given in the path (absolute
      or relative to Current Working Directory).

      By overloading this method, subclasses may create method loaders for
      non-disk resources (i.e. interface virtual file systems on compressed
      files or on networks).

      The method may invoke the error handler to signal errors, and will
      return 0 in case the file can't be opened for reading.

      An optional file type is provided to give the subclasses an hint on
      how to configure the stream for the final user.

      \param path the path to file
      \param type the file type that should be opened (t_none meaning unknown / generic).
      \return a valid input stream or 0 on error.
   */
   virtual Stream *openResource( const String &path, t_filetype type = t_none );

   /** Scan for files that may be loaded.

      This function searches for files that may be loaded under the loader rules.
      A "name" string is provided as parameter; the name may either be a
      logical module name or a parth of a file name or relative path to be
      searched in the target filesystem; the parameter \b isPath will determine
      how the \b name parameter should be considered.

      If the function is succesful, the function returns a file type and sets
      the \b found accordingly to the path that can be used to open the file.

      The algorithm searches for binary modules, and if they are not found,
      for native "fam" modules. If the \b searchSources parameter is true,
      then the function should return sources instead of .fam modules, if
      they are found.

      In case the \b name parameter is a path, the function will check for
      the file to have an extension; if the extension is a known one, it
      won't automatically add known other known extensions, and will only
      perform a scan within the knwon directory. Overloaded method should
      respect this behavior.

      This function is meant to allow subclassess to determine file types
      on their own  will; this method tell sources, binary module and
      Falcon native modules. This basic version of the method will try
      to open the file for reading to determine their type. It will
      also close them before returning.
      \param name partial path to a file or logical module name
      \param isPath true will indicate that name is a partial path
      \param scanForType set to t_none to search for any kind of file, or to a specific type to limit the search
      \param found if a file can be found, a complete path to the file
      \param searchSources true to include sources in the search
      \return on success, returns the file type for the \b found parameter, else returns t_none.
   */

   virtual t_filetype scanForFile( const String &name, bool isPath, t_filetype scanForType,
         String &found,
         bool searchSources = false );

   /** Determine file types.
      This method tries to determine the type of a file given a path.+

      This function is meant to allow subclassess to determine file types
      on their own  will; this method tell sources, binary module and
      Falcon native modules. This basic version of the method will try
      to open the file for reading to determine their type. It will
      also close them before returning.

      \param path the file to be tested (absolute / relative to cwd)
      \return the type of file or t_none if it can't be determined.
   */

   virtual t_filetype fileType( const String &path );

public:

   /** Creates a module loader.
      As default, the current directory is included in the path. If this is not desired, use
      ModuleLoader( "" ) as a constructor.
   */
   ModuleLoader():
    m_errhand(0),
    m_acceptSources( false )
   {
      m_path.pushBack( new String(".") );
   }

   ~ModuleLoader();

   /** Creates a module loader with a given path.
      The path is in the FALCON format; path are separated by semicolons (;). If the system uses
      disk specification, falcon is able to understand a single letter, a colon and a slash as
      a disk specification. I.e.

      \code
         ".;C:/projects;d:/modules";
      \endcode

      will be processed as ".", "C:\projects" and "D:\modules" under systems using disk specifications.

      The current directory is \b not included in the default path. If this is desired, add it
      as a single "." entry.
   */
   ModuleLoader( const String &path );

   /** Changes the search path used to load modules by this module loader

      This method changes the search specification path used by this module
      loader with the one provided the parameter to the
      search path of those already known. Directory must be expressed in Falcon
      standard directory notation ( forward slashes to separate subdirectories).
      If the path contains more than one directory
      they must be separated with a semicomma; in example:

      \code
         modloader.addSearchPath( "../;/my/modules;d:/other/modules" );
      \endcode


      \see addSearchPath getSearchPath
      \see ModuleLoader
      \param path the new module search path
   */
   void setSearchPath( const String &path );

   /** Add standard system Falcon module paths to the search path of this module loader.

      By default, Falcon system module paths are not added to newly created Module Loader.
      This is because an embedding application may wish to load its own version of the modules
      from somwhere else.

      This method appends the path where Falcon is installed on this system, to the search
      path of this module loader. The added path will be searched after the ones that have
      already been added to this loader. To give system Falcon libraries and modules higher
      priority, call this method before adding your application paths (including ".").

      If Falcon is not installed on this system, this method will have no effect.
   */
   void addFalconPath();

   /** Adds one or more directories to the module loader search path.
      This method appends the search specification path passed as a parameter to the
      search path of those already known. Directory must be expressed in Falcon
      standard directory notation ( forward slashes to separate subdirectories).
      If the path contains more than one directory
      they must be separated with a semicomma; in example:

      \code
         modloader.addSearchPath( "../;/my/modules;/other/modules" );
      \endcode

      The directories will be added with a priority lower than the currently searched ones;
      that is, they will be searched after the ones that have been previously added are
      searched.

      \see setSearchPath getSearchPath
      \param path the search specification to be added to the path
   */
   void addSearchPath( const String &path );

   /** Adds a single directory to the module loader path, with higher priority.

      This method adds a directory with a priority higher than the ones already defined.

      \note Don't use ";" separated paths here; just call this method once for each
      path.

      \param directory the directory to be added to the path
   */
   void addDirectoryFront( const String &directory )
   {
      if ( directory != "" )
         m_path.pushFront( new String( directory ) );
   }

   /** Adds a single path specification to the module loader path, with lower priority.
      This method adds a directory with a priority lower than the ones already defined.

      \note Don't use ";" separated paths here; just call this method once for each
      path.

      \param directory the directory to be added to the path
   */
   void addDirectoryBack( const String &directory )
   {
      if ( directory != "" )
         m_path.pushBack( new String( directory ) );
   }

   /** Loads a module by its name.
      This function scan the directories in the path for the given module,
      be it a binary module (loadable object or dll) or a falcon native
      module ("fam" format).

      If the extension is not given, the loader will try to load first the file
      as it is given (without extension), trying to detect it's type. Then, it
      will add first the system loadable module extension (i.e. ".so" or ".dll"),
      and if a file with that name cannot be found, it will try by adding ".fam".

      The process terminates when a module can be loaded, or when the search path
      is exhausted.

      In case of error, it will be reported to the error manager and 0 will be returned.

      Subclasses may filter modules to prevent the code to load unwanted modules, change
      default modules with application specific ones or other mangling things.
      \note On failure, the module loader will post an error to its error handler, if
      it has been provided.

      Once found a suitable candidate, this method calls either loadModule() or
         loadBinModule() to load the module from the filesystem.

      \note On success, the returned module will have its logical name and path
      set accordingly to the module_name parameter and the path where the module
      has been found.

      \param module_name the name of the module to search for
      \param parent_name the name of the module that is asking for this module to be loaded
      \return 0 on failure or a newly allocated module on success.
   */
   virtual Module *loadName( const String &module_name, const String &parent_module = "" );

   /** Loads a module by its path.
      This method loads directly a module. If the \b scan parameter is given \b and
      the path is relative, then the module is searched in the modloader path. Current
      directory is not explicitly searched unless it is present in the loader search
      path.

      If the path is relative and \b scan is false, the path will be considered relative
      to current working directory.

      If the type of file is not given (set to t_none) the method tries to dentermine
      the type of the file. If it's given, it will ignore the file type detection and
      will pass an opened stream directly to the loadModule() or loadBinModule() method.

      This implementation will raise an error if t_source is explicitly provided as a
      type or if the target file is detected as a source.

      On load, the logical module name will be set to the file part of the path. However,
      it may be changed after loading.

      \note on failure, the module loader will post an error to its error handler, if
      it has been provided.

      \param module_path the relative or absolute path of the file to load
      \param type the type of the file to be loaded, or t_none to autodetect
      \param scan if module_path is relative, set to true to allow scanning of the modloader path
      \return a valid module on success.
   */
   virtual Module *loadFile( const String &module_path, t_filetype type=t_none, bool scan=false );

   /** Loads a Falcon precompiled native module from a given path
      This is a shortcut that simply opens the module file and loads it
      through loadModule( Stream * ). After a succesful load, the module
      path and logical name will be filled accordingly.

      \param file  A direct relative or absolute path to an openable serialized source.
      \return 0 on failure or a newly allocated module on success.
   */
   virtual Module *loadModule( const String &file );

   /** Loads a Falcon precompiled native module from the input stream.
      This function tries to load a Falcon native module. It will
      detect falcon module mark ('F' and 'M'), and if successful
      it will recognize the module format, and finally pass the
      stream to the correct module loader for the version/subversion
      that has been detected.

      On success, a new memory representation of the module, ready
      for linking, will be returned. On failure, 0 will be returned
      and the error handler will be called with the appropriate
      error description.

      \note after loading, the caller must properly set returned module
      name and path.

      \param input An input stream from which a module can be deserialized.
      \return 0 on failure or a newly allocated module on success.
   */
   virtual Module *loadModule( Stream *input );

   /** Loads a Falcon source from a given path.
      This is a shortcut that simply opens the module through openResource()
      file and loads it through loadSource( Stream * ). After a succesful load,
      the module path and logical name will be filled accordingly.

      \param file A direct relative or absolute path to an openable source.
      \return 0 on failure or a newly allocated module on success.
   */
   virtual Module *loadSource( const String &file );

   /** Loads a source stream.
      In the base class, this module returns always 0 raising an error.
      This method is meant to be overloaded by subclasses that accepts
      sources, i.e. by integrating with the compiler.

      \note after loading, the caller must properly set returned module
      name and path.

      \param input an opened input stream delivering the source to be parsed.
      \param file complete path to the loaded file. Useful i.e. to be set in the compiler.
      \return 0 on failure or a newly allocated module on success.
   */
   virtual Module *loadSource( Stream *input, const String &file );


   /** Loads a binary module by realizing a system dynamic file.

      The method calls the dll/dynamic object loader associated with this
      module loader. The dll loader usually loads the dynamic objects
      and calls the startup routine of the loaded object, which should
      return a valid Falcon module instance.

      This method then fills the module path and logical name accordingly
      with the given parameter.

      Special strategies in opening binary modules may be implemented
      by subclassing the binary module loader.

      \note after loading, the caller must properly set returned module
      name and path.

      \param module_path the relative or absolute path.
      \return 0 on failure or a newly allocated module on success.
   */
   virtual Module *loadBinaryModule( const String &module_path );

   void raiseError( int code, const String &expl );
   void raiseError( int code )
   {
      raiseError( code, "" );
   }


   /** Get the search path used by this module loader.
      \param target a string where the path will be saved.
   */
   void getSearchPath( String &target ) const;

   /** Get the search path used by this module loader.

      \return the search path that is searched by this module loader.
   */
   String getSearchPath() const
   {
      String temp;
      getSearchPath( temp );
      return temp;
   }

   void errorHandler( ErrorHandler *h ) { m_errhand = h; }
   ErrorHandler *errorHandler() const { return m_errhand; }
};

}

#endif
/* end of flc_modloader.h */
