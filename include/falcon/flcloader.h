/*
   FALCON - The Falcon Programming Language.
   FILE: flcloader.h
   $Id: flcloader.h,v 1.8 2007/07/27 12:03:09 jonnymind Exp $

   Advanced module loader for Falcon programming language.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar nov 22 2005
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Advanced module loader - header
*/

#ifndef FLC_FLCLOADER_H
#define FLC_FLCLOADER_H

#include <falcon/modloader.h>
#include <falcon/compiler.h>


namespace Falcon {

/** Advanced module loader.
   The basic class used for loading falcon modules, ModuleLoader,
   allows only to retrieve from disk already existing pre-compiled
   falcon modules, or binary libraries. This is a desirable behavior
   in many embedded applications, and thus the basic module loader
   is contained in the VM library for the embedders to use or
   extend.

   However, when Falcon is seen as a standalone scripting language,
   it is desirable that it also compiles on the fly modules written
   in Falcon language, or chose between the option to compile them
   on the fly or to use already existing modules if their timestamp
   is newer than the source they come from.

   Instead of just implementing this feature in the Falcon standalone
   interpreter, this class extends the module loader and interfaces
   the compiler to provide this functionality. So, embedding applications
   wishing to include file compilation facilities may use this class that
   automatically scans for sources and compile them on "load" requests
   in master modules.

   By passing this module loader to the RunTime object, whenever a
   module will declare a "load" directive, the runtime will rely on
   this object to extract a module from a source, if this action
   is consistent.

*/

class FALCON_DYN_CLASS FlcLoader: public ModuleLoader
{
   bool m_alwaysRecomp;
   bool m_compMemory;
   bool m_viaAssembly;
   bool m_saveModule;
   bool m_sourceIsAssembly;
   bool m_saveMandatory;
   bool m_detectTemplate;
   bool m_forceTemplate;

   bool m_delayRaise;
   uint32 m_compileErrors;

   Compiler m_compiler;

   Module *compile( const String &path );
   t_filetype searchForModule( String &final_name );
   t_filetype checkForModuleAlreadyThere( String &final_name );

   String m_srcEncoding;
protected:

   /** Overloaded from ModuleLoader.
      If the provided filetype is t_source, a source encoder is provided to encapsulate
      the stream.
   */
   virtual Stream *openResource( const String &path, t_filetype type = t_none );

public:
   FlcLoader( const String &path );

   /** Ignore Source accessor.
      \return true if the Module Loader must load only pre-compiled modules, false otherwise.
   */
   bool ignoreSources() const { return ! m_acceptSources; }

   /** Always recompile accessor.
      \return true if source modules must always be recompiled before loading.
   */
   bool alwaysRecomp() const { return m_alwaysRecomp;}

   void ignoreSources( bool mode ) { m_acceptSources = ! mode; }
   void alwaysRecomp( bool mode ) { m_alwaysRecomp = mode; }

   void compileInMemory( bool ci ) { m_compMemory = ci; }
   bool compileInMemory() const { return m_compMemory; }

   void compileViaAssembly( bool ca ) { m_viaAssembly = ca; }
   bool compileViaAssembly() const { return m_viaAssembly; }

   void saveModules( bool t ) { m_saveModule = t; }
   bool saveModules() const { return m_saveModule; }

   void sourceEncoding( const String &name ) { m_srcEncoding = name; }
   const String &sourceEncoding() const { return m_srcEncoding; }

   void sourceIsAssembly( bool b ) { m_sourceIsAssembly = b; }
   bool sourceIsAssembly() const { return m_sourceIsAssembly; }

   void delayRaise( bool setting ) { m_delayRaise = setting; }
   bool delayRaise() const { return m_delayRaise; }

   void saveMandatory( bool setting ) { m_saveMandatory = setting; }
   bool saveMandatory() const { return m_saveMandatory; }

   void detectTemplate( bool bDetect ) { m_detectTemplate = bDetect; }
   bool detectTemplate() const { return m_detectTemplate; }

   void compileTemplate( bool bCompTemplate ) { m_forceTemplate = bCompTemplate; }
   bool compileTemplate() const { return m_forceTemplate; }

   /** return last compile errors. */
   uint32 compileErrors() const { return m_compileErrors; }

   /** Load a source.
      Unless the alwaysRecomp() option is set, the target directory is
      scanned for a file with the same name of the source, with .fam
      extension, and if that one is found and it has a timestamp newer
      than the source, that one is loaded instead.

      If sourceIsAssembly() is true, then the input file is considered an
      assembly source and compiled via the assembler.

      \param file a complete (relative or absolute) path to a source to be compiled.
      \return a valid module on success, 0 on failure (with error risen).
   */
   virtual Module *loadSource( const String &file );

   /** Compile the source from an input stream.
       Actually an alias for loadSource. The input stream must be already
       correctly transcoded.
   */
   virtual Module *loadSource( Stream *in, const String &path );

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

/* end of flcloader.h */
