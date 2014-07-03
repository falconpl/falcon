/*
   FALCON - The Falcon Programming Language.
   FILE: modloader.h

   Module loader and reference resolutor.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 01 Aug 2011 11:45:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_MODLOADER_H_
#define _FALCON_MODLOADER_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/uri.h>
#include <falcon/pstep.h>
#include <falcon/stream.h>

namespace Falcon
{

class ModSpace;
class ModCompiler;
class FAMLoader;
class DynLoader;
class Transcoder;
class Error;
class VMContext;

/** Module loader and reference resolutor.

 The Module Loader is a helper class that searches for a module file
 given a logical module name or an URI, and then either compiles, deserializes
 or loads it as a native dynamic shared object (dll/so/dylib etc.), and
 serves it as a Falcon::Module entity.

 On success, the module is loaded and ready to be added to a ModSpace.
 Dependencies of the module are not resolved.

 This class uses instances of helper classes to access the required resources;
 those classes can be used directly if the location and type of the module
 are known in advance. They are:

 - ModCompiler: compiles a module from an open TextReader.
 - FAMLoader: loads a precompiled falcon module from an open DataReader.
 - DynLoader: loads a native shared object providing a Falcon Moduler.

 To compile a source code directly (i.e. from an open stream, or from
 inside a String), use the ModCompiler class.

 To load a moduel
 */
class FALCON_DYN_CLASS ModLoader
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
      e_mt_none,
      /** The module is a source. */
      e_mt_source,
      /** The module is a Falcon Templage Document. */
      e_mt_ftd,
      /** The module is a falcon FAM module. */
      e_mt_vmmod,
      /** The module is a native .so/.dll. */
      e_mt_binmod,
      /** Special file type.
          Try to determine the type of the file,
          but in case it cannot be determined, defaults to source.
      */
      t_defaultSource
   } t_modtype;

   /** Creates the ModLoader.
    \param ctx The context where the loading occurs.
    \param owner The module space owning this loader.
    \param mc An optional previously created and configured ModCompiler.
    \param faml An optional previously created and configured FamLoader.
    \param dld An optional previously created and configured DynLoader.

    If not given, the module compiler used by this ModLoader will be
    created internally.

    The ownership of the ModCompiler stays on this instance; the compiler
    willl be destroyed with it.

    This constructor sets the path to the default ("." + system falcon load path)
    */
   ModLoader( ModSpace* owner, ModCompiler* mc = 0, FAMLoader* faml=0, DynLoader* dld=0 );

   /** Creates a module loader with a given path.
    \param path The path where modules will be searched for.
    \param ctx The context where the loading occurs.
    \param owner The module space owning this loader.
    \param mc An optional previously created and configured ModCompiler.
    \param faml An optional previously created and configured FamLoader.
    \param dld An optional previously created and configured DynLoader.

    The path is in the FALCON format; path are separated by semicolons (;). If the system uses
      disk specification, falcon is able to understand a single letter, a colon and a slash as
      a disk specification. I.e.

      \code
         ".;/C:/projects;d:/modules";
      \endcode

      will be processed as ".", "C:\projects" and "D:\modules" under systems using disk specifications.

      The current directory is \b not included in the default path. If this is desired, add it
      as a single "." entry.
   */
   ModLoader( const String &path, ModSpace* owner, ModCompiler* mc = 0, FAMLoader* faml=0, DynLoader* dld=0 );

   ~ModLoader();


   /** Gets the compiler used by this loader. */
   ModCompiler* compiler() const { return m_compiler; }

   /** Gets the pre-compiled module loader used by this loader. */
   FAMLoader* famLoader() const { return m_famLoader; }

   /** Gets the dynamic native module loader used by this loader. */
   DynLoader* dynLoader() const { return m_dynLoader; }

   /** Loads a module through its logical name.
    \param path The path of the module.
    \param type Detect the type of the resource to be loaded or provide a
    specific type.
    \param loader The module that originated the request, if any. Used to relativize file names.
    \return 0 If the module could not be found
    \throw Error* or appropriate error subclass in case of other errors.

    \note The name of the module must be already de-relativized by adding
    the proper parent module names in case of self.xxx or .xxx module naming
    convnetion.
    */
   bool loadName( VMContext* tgtctx, const String& name, t_modtype type=e_mt_none, Module* loader = 0 );

   /** Loads a module through its physical path.
    \param path The path of the module.
    \param name Logical name to be assigned to the module.
    \param type Detect the type of the resource to be loaded or provide a
    specific type.
    \param bScan if true and the path is relative, the module will be searched
    in the search path; else, it will be searched only relatively to the
    current directory.
    \param loader The module that originated the request, if any. Used to relativize file names.
    \return 0 If the module could not be found
    \throw Error* or appropriate error subclass in case of other errors.

    If @b name is not given (is an empty string), the name will be automatically calculated using
    the module path. If it is given, the name will be assigned to the loaded module, ingoring
    its actual path. The name will be eventually adapted using the loader name if it starts with a dot
    or it contains the self keyword.
    */
   bool loadFile( VMContext* tgtctx, const String& name, const String& path, t_modtype type=e_mt_none, bool bScan = true, Module* loader = 0 );

   /** Loads a module through its physical path.
    \param uri The uri of the module.
    \param type Detect the type of the resource to be loaded or provide a
    \param name Logical name to be assigned to the module.
    specific type.
    \param bScan if true and the path is relative, the module will be searched
    \param loader The module that originated the request, if any. Used to relativize file names.
    \return 0 If the module could not be found
    \throw Error* or appropriate error subclass in case of other errors.

    This version uses an URI instead of a String.
    */
   bool loadFile( VMContext* tgtctx,  const String& name, const URI& uri, t_modtype type=e_mt_none, bool bScan = false, Module* loader = 0 );
   bool loadMem( VMContext* tgtctx,  const String& name, Stream* script, const String& path = "", t_modtype type = e_mt_source );
   //============================================================
   // Compilation process setting
   //

   /** Should compilation process save the precompiled modules? */
   typedef enum {
      /** Never save the precompiled modules. */
      e_save_no,
      /** Try to save precompiled modules and ignore errors. */
      e_save_try,
      /** Save precompiled modules and raise an error if not possible. */
      e_save_mandatory
   } t_save_pc;

   /** Consider input source as ftd? */
   typedef enum {
      /** Sources are never FTD */
      e_ftd_ignore,
      /** Consider FTD sources depedning on source extension. */
      e_ftd_check,
      /** All sources are FTD */
      e_ftd_force
   } t_check_ftd;

   /** Use sources when in doubt between a source and a precompiled? */
   typedef enum {
      /** If sources are newer than FAM, use sources. */
      e_us_newer,
      /** Always use soucrces. */
      e_us_always,
      /** Ignore sources, and use FAM only. */
      e_us_never
   } t_use_sources;

   /** Check if this ModSpace will save precompiled sources.*/
   t_save_pc savePC() const { return m_savePC; }
   /** Set how this ModSpace saves precompiled sources. */
   void savePC( t_save_pc value ) { m_savePC = value; }

   /** Check if this ModLoader searches for FTD sources. */
   t_check_ftd checkFTD() const { return m_checkFTD; }
   /** Sets how this ModLoader searches for FTD sources. */
   void checkFTD( t_check_ftd value ) { m_checkFTD = value; }

   /** Indicates how this ModLoader consider source files.*/
   t_use_sources useSources() const { return m_useSources; }
   /** Alters how this ModLoader consider source files.*/
   void useSources( t_use_sources value ) { m_useSources = value; }

   /** Checks if this ModLoader tries to save FAM modules on remote devices.*/
   bool saveRemote() const { return m_saveRemote; }
   /** Sets if this ModLoader tries to save FAM modules on remote devices.*/
   void saveRemote( bool value ) { m_saveRemote = value; }

   /** Returns the current extension for FTD files. */
   const String& ftdExt() const { return m_ftdExt; }

   /** Changes the extension for FTD files. */
   void ftdExt( const String& value ) { m_ftdExt = value; }


   /** Returns the current extension for precompiled files. */
   const String& famExt() const { return m_famExt; }

   /** Changes the extension for FTD files. */
   void famExt( const String& value ) { m_famExt = value; }

   /** Sets the given source file encoding.
    \param encName One of the available ISO/POSIX encoding names.
    \return True if the encoding is known, false if the encoding is not available.

    If the loaded modules are sources, this setting will be used to
    determine which encoding should be used.

    \note Set the encoding name to "C" to use no text transcoding.

    \TODO Add auto-detection.
    */
   bool sourceEncoding( const String& encName );

   const String& sourceEncoding() const { return m_encName; }

   //=============================================================
   // Search path manipulation
   //

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

      The path can contain complete URIs. When an URI is not given, the local
      filesystem will be used.

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
   void addDirectoryFront( const String &directory );

   /** Adds a single path specification to the module loader path, with lower priority.
      This method adds a directory with a priority lower than the ones already defined.

      \note Don't use ";" separated paths here; just call this method once for each
      path.

      \param directory the directory to be added to the path
   */
   void addDirectoryBack( const String &directory );

   /** Returns the currently set search path. */
   const String& getSearchPath() const;

   //==============================================================
   // Utility functions
   //

   /** Converts a module physical location into a logical name.
      \param path Relative or absolute top-path.
      \param modFile the path to a possible falcon module
      \param modNmae the possible falcon module name
   */
   static void pathToName( const URI &prefix, const URI &modFile, String &modName );


private:
   class Private;
   Private* _p;

   ModSpace* m_owner;
   ModCompiler* m_compiler;
   FAMLoader* m_famLoader;
   DynLoader* m_dynLoader;

   t_save_pc m_savePC;
   t_check_ftd m_checkFTD;
   t_use_sources m_useSources;

   bool m_saveRemote;

   String m_ftdExt;
   String m_famExt;
   mutable String m_path;

   String m_encName;
   Transcoder* m_tcoder;

   void init ( const String &path, ModSpace* ms, ModCompiler* mc, FAMLoader* faml, DynLoader* dld );

   t_modtype checkFile_internal( const URI& uri, t_modtype type, URI& foundUri );
   void load_internal( VMContext* tgtctx, const String& name, const String& prefixPath, const URI& uri, t_modtype type );
   void saveModule_internal( VMContext* tgtctx, Module* module, const URI& uri, const String& modName );
   Error* makeError( int code, int line, const String &expl="", int fsError=0 );

   class FALCON_DYN_CLASS PStepSave: public PStep
   {
   public:
      PStepSave( ModLoader* owner ): m_owner( owner ) {
         apply = apply_;
      }
      virtual ~PStepSave() {};
      virtual void describeTo( String& str ) const { str = "PStepSave"; }

      //Need to do something about this
    ModLoader* m_owner;

   private:
      static void apply_( const PStep* self, VMContext* ctx );

   };
   PStepSave m_stepSave;

   friend class PStepSave;
};

}

#endif	/* _FALCON_MODLOADER_H_ */

/* end of modloader.h */
