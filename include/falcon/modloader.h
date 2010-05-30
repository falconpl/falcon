/*
   FALCON - The Falcon Programming Language.
   FILE: flc_modloader.h

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar ago 3 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FLC_MODLOADER_H
#define FLC_MODLOADER_H

#include <falcon/common.h>
#include <falcon/error.h>
#include <falcon/string.h>
#include <falcon/basealloc.h>
#include <falcon/compiler.h>

namespace Falcon {

class Module;
class Stream;
class URI;
class FileStat;
class VFSProvider;

/** Module Loader support.

   This class enables embedding applications and falcon VM (and thus, Falcon scripts)
   to load modules.
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
      /** The module is a Falcon Templage Document. */
      t_ftd,
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


protected:
   Module *loadModule_ver_1_0( Stream *in );

   bool m_alwaysRecomp;
   bool m_compMemory;
   bool m_saveModule;
   bool m_saveMandatory;
   bool m_detectTemplate;
   bool m_forceTemplate;
   bool m_delayRaise;
   bool m_ignoreSources;
   bool m_saveRemote;
   uint32 m_compileErrors;

   Compiler m_compiler;
   String m_srcEncoding;
   bool m_bSaveIntTemplate;

   Module *compile( const String &path );
   t_filetype searchForModule( String &final_name );
   t_filetype checkForModuleAlreadyThere( String &final_name );


   /** Required language during load. */
   String m_language;

   Module *loadModule_select_ver( Stream *in );

   /** Discovers the module name given a complete file path.
      \param path the path to a possible falcon module
      \param modNmae the possible falcon module name
   */
   static void getModuleName( const String &path, String &modName );

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
      Utility funciton searching for one possible file in a directory.

      Tries all the possible system extensions (applied directly on origUri)
      and searches for a matching file. If one is found, true is returned and
      type and fs are filled with the item module-type and its data.

      Priority is scan file types are ftd, fal, fam and .so/.dll/.dylib.

      The function doesn't check for validity of the given file, but only
      for its existence.

      \return on success, returns true.
   */
   bool scanForFile( URI &origUri, VFSProvider*, t_filetype &type, FileStat &fs );

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

   /** Try to load the language table for the given module. */
   bool applyLangTable( Module *mod, const String &file_path );

public:

   /** Creates a module loader.
      As default, the current directory is included in the path. If this is not desired, use
      ModuleLoader( "" ) as a constructor.
   */
   ModuleLoader();
   ModuleLoader( const ModuleLoader &other );

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

   virtual ~ModuleLoader();

   virtual ModuleLoader *clone() const;


   /** Changes the search path used to load modules by this module loader

      This method changes the search specification path used by this module
      loader with the one provided the parameter to the
      search path of those already known. Directory must be expressed in Falcon
      standard directory notation ( forward slashes to separate subdirectories).
      If the path contains more than one directory
      they must be separated with a semicomma; for example:

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
      they must be separated with a semicomma; for example:

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
      This function scan the directories in the path for a matching module
      name. The logical module name, which can be logically related with a
      parent module (as in the case of "load self.submodule") is used to
      determine possible phisical file names within the given search path
      and filesystems.

      Once a suitable file is found, loadFile() is called with the
      appriopriate path and type parameters. The loadFile() method has its own
      precompiled-versus-source resolution logic. As this function is just a
      front-end to resolve logical name into potential physical names, this function
      follows the same compile-or-load logic as loadFile().

      \note On success, the returned module will have its physical name and path
      set accordingly to the module_name parameter and the path where the module
      has been found.

      If a suitable file is found, but fails to load (beynd the recovery logic
      as stored in loadFile()), the search is interrupted and an error is raised.

      \param module_name the name of the module to search for
      \param parent_name the name of the module that is asking for this module to be loaded
      \return newly allocated module on success.
      \throw Error or appropriate subclass on error.
   */
   virtual Module *loadName( const String &module_name, const String &parent_module = "" );

   /** Loads a module by its path.

      This method loads directly a module. If the \b scan parameter is given \b and
      the path is relative, then the module is searched in the modloader path. Current
      directory is not explicitly searched unless it is present in the loader search
      path.

      If the path is relative and \b scan is false, the path will be considered relative
      to current working directory. If it's true, a relative path would be searched
      across all the paths in turn, that doesn't necesarily include the current
      working directory.

      If the type of file is not given (set to t_none) the method tries to dentermine
      the type of the file. If it's given, it will ignore the file type detection and
      will pass an opened stream directly to the loadSource(), loadModule() or loadBinModule()
      method.

      In case the module is determined to be a source (or an FTD), this method scans for
      a file in the same directory with a .fam extension, and checks if the module can
      is newer and can be loaded (unless alwaysRecomp() is true, In this case, if the
      .fam module fails to load, then the the error is discarded and the program
      continues trying to compile the original source. If trying to use a .fam is not
      desired, either set alwaysRecomp() or call directly loadSource().

      Conversely, if the file is determined to be a .fam module, a source with the same
      name but with newer timestamp is searched, and eventually compiled if found. If
      this is not wanted, either call directly loadModule() or set ignoreSources() to
      true.

      This implementation will raise an error if t_source is explicitly provided as a
      type or if the target file is detected as a source.

      On load, the logical module name will be set to the file part of the path. However,
      it may be changed after loading.

      \note on failure, the module loader will post an error to its error handler, if
      it has been provided.

      \param module_path the relative or absolute path of the file to load. The file is not
         URI encoded; to load URI encoded filenames, pass directly the URI file.

      \param type the type of the file to be loaded, or t_none to autodetect
      \param scan if module_path is relative, set to true to allow scanning of the modloader path
      \return a valid module on success.
      \throw Error or appropriate subclass on error.
   */
   virtual Module *loadFile( const String &module_path, t_filetype type=t_none, bool scan=false );

   /** Loads a module by its URI.
      \see Module *loadFile( const String &module_path, t_filetype type=t_none, bool scan=false );
      \param module_URI the relative or absolute path of the file to load
      \param type the type of the file to be loaded, or t_none to autodetect
      \param scan if module_path is relative, set to true to allow scanning of the modloader path
      \return a valid module on success.
      \throw Error or appropriate subclass on error.
   */
   virtual Module *loadFile( const URI &module_URI, t_filetype type=t_none, bool scan=false );

   /** Loads a Falcon precompiled native module from the input stream.

      This function tries to load a Falcon native module. It will
      detect falcon module mark ('F' and 'M'), and if successful
      it will recognize the module format, and finally pass the
      stream to the correct module loader for the version/subversion
      that has been detected.

      On success, a new memory representation of the module, ready
      for linking, will be returned.

      \note after loading, the caller must properly set returned module
      name and path.

      \param input An input stream from which a module can be deserialized.
      \return a newly allocated module on success.
      \throw Error or appropriate subclass on error.
   */
   virtual Module *loadModule( Stream *input );

   /** Loads a Falcon precompiled native module.

      Front end for loadModule(Stream).

      This function sets the module name and path accordingly to the \b file
      parameter. The caller may know better and reset the module logical
      name once a valid module is returned.

      \note This method doesn't set the module language table.

      \param path A path from which to load the module.
      \return a newly allocated module on success.
      \throw Error or appropriate subclass on error.

      TODO: make virtual
   */
   Module *loadModule( const String &file );

   /** Load a source.

      Tries to load a file that is directly considered a source file. This is just
      a front-end to loadSource( Stream*,const String &, const String & ).

      This function sets the module name and path accordingly to the \b file
      parameter. The caller may know better and reset the module logical
      name once a valid module is returned.

      \note This method doesn't set the module language table.

      \param file a complete (relative or absolute) path to a source to be compiled.
      \return a valid module on success, 0 on failure (with error risen).
      \throw Error or appropriate subclass on error.
   */
   virtual Module *loadSource( const String &file );

   /** Compile the source from an input stream.

      This function sets the module name and path accordingly to the \b file
      parameter. The caller may know better and reset the module logical
      name once a valid module is returned.

       \note Notice that this function doesn't load the translation
       table.

      Also, this function tries to save the generated module in a .fam file,
      if saveModules is set to true; if saveMandatory is also true, the function
      will raise an error if the compiled module can't be properly saved.

       \throw Error or appropriate subclass on error.
       \param in the file from which to load the file.
       \param uri the complete URI of the source file from which the stream is open,
       \param modname logical name of the module.
   */
   virtual Module *loadSource( Stream *in, const String &uri, const String &modname );

   /** Loads a binary module by realizing a system dynamic file.

      The method calls the dll/dynamic object loader associated with this
      module loader. The dll loader usually loads the dynamic objects
      and calls the startup routine of the loaded object, which should
      return a valid Falcon module instance.

      This method then fills the module path and logical name accordingly
      with the given parameter.

      Special strategies in opening binary modules may be implemented
      by subclassing the binary module loader.

      This function sets the module name and path accordingly to the \b file
      parameter. The caller may know better and reset the module logical
      name once a valid module is returned.

      \note This method doesn't set the module language table.

      \param module_path the relative or absolute path.

      \return newly allocated module on success.
      \throw Error or appropriate subclass on error.
   */
   virtual Module *loadBinaryModule( const String &module_path );

   void raiseError( int code, const String &expl, int fsError=0 );

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

   /** Save international templates for loaded modules.
      If this option is set to true, and if the loaded modules
      have international strings, then a template for the
      internationalization file will be saved.
   */
   void saveIntTemplates( bool mode /*, bool force=false */ )
   {
      m_bSaveIntTemplate = mode;
   }

   /** Sets the language required to modules during load.
      This informs the module loader that the owner wishes the string table
      of the loaded module configured for the given language.

      If the loaded module doesn't declare itself to be written in the
      desired language, the module loader will try to load \b modulename.ftr
      binary file, get the table for the desired language and change the
      strings according to the translation table before returning it to
      the caller.

      In case of failure, the original string table will be left untouched.

      Language names are the ISO language names in 5 characters: xx_YY.

      Setting the language to "" disables this feature.

      \param langname the name of the language that should be loaded.
   */
   void setLanguage( const String &langname ) { m_language = langname; }

   /** Returns the translation language that is searched by this module loader.
   */
   const String &getLanguage() const { return m_language; }

   /** Load a determined language table directly into the module.
      On success, the language table of the module and it's declared language
      are changed.
      \return true on success.
   */
   bool loadLanguageTable( Module *module, const String &language );

   /** Ignore Source accessor.
      \return true if the Module Loader must load only pre-compiled modules, false otherwise.
   */
   bool ignoreSources() const { return m_ignoreSources; }

   /** Always recompile accessor.
      \return true if source modules must always be recompiled before loading.
   */
   bool alwaysRecomp() const { return m_alwaysRecomp;}

   void ignoreSources( bool mode ) { m_ignoreSources = mode; }
   void alwaysRecomp( bool mode ) { m_alwaysRecomp = mode; }

   void compileInMemory( bool ci ) { m_compMemory = ci; }
   bool compileInMemory() const { return m_compMemory; }

   void saveModules( bool t ) { m_saveModule = t; }
   bool saveModules() const { return m_saveModule; }

   void sourceEncoding( const String &name ) { m_srcEncoding = name; }
   const String &sourceEncoding() const { return m_srcEncoding; }

   void delayRaise( bool setting ) { m_delayRaise = setting; }
   bool delayRaise() const { return m_delayRaise; }

   void saveMandatory( bool setting ) { m_saveMandatory = setting; }
   bool saveMandatory() const { return m_saveMandatory; }

   void detectTemplate( bool bDetect ) { m_detectTemplate = bDetect; }
   bool detectTemplate() const { return m_detectTemplate; }

   void compileTemplate( bool bCompTemplate ) { m_forceTemplate = bCompTemplate; }
   bool compileTemplate() const { return m_forceTemplate; }

   /** Tells if this modloader should save .fam on remote filesystems.
      By default, if a source file is loaded from a remote filesystem,
      the module loader doesn't try to save a .fam serialized version
      of the module besides the source.

      You can set this to true to force a try to store modules also
      on remote filesystems.
      \param brem true to try to save .fam files on remote filesystems.
   */
   void saveRemote( bool brem ) { m_saveRemote = brem; }

   /** Tells wether this loader tries to save .fam on remote filesystems.
   \see saveRemote( bool )
   */
   bool saveRemote() const { return m_saveRemote; }

   /** return last compile errors. */
   uint32 compileErrors() const { return m_compileErrors; }

   /** Return the compiler used by this module loader.
      This object can be inspected, or compiler options can be set by the caller.
      \return a reference of the compiler used by the loader.
   */
   const Compiler &compiler() const { return m_compiler; }

   /** Return the compiler used by this module loader (non const).
      This object can be inspected, or compiler options can be set by the caller.
      \return a reference of the compiler used by the loader.
   */
   Compiler &compiler() { return m_compiler; }
};

}

#endif
/* end of modloader.h */
