/*
   FALCON - The Falcon Programming Language.
   FILE: dynloader.h

   Native shared object based module loader.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 01 Aug 2011 16:07:56 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_DYNLOADER_H_
#define _FALCON_DYNLOADER_H_

#include <falcon/setup.h>
#include <falcon/refcounter.h>
#include <falcon/string.h>

namespace Falcon
{

class Module;

/** Utility directly loading a dynamic library.
 * Useful for Falcon users, as it ports across platforms.
 *
 */
class DynLibrary
{
public:
   /** Creates a dynamic library that needs to be open */
   DynLibrary();

   /** Creates and open a dynamic library.
    * \param path Path to a local filesystem loadable module.
    * \throw IoError on error.
    */
   DynLibrary( const String& path );

   /** Creates and open a dynamic library.
    * \param path Path to a local filesystem loadable module.
    * \throw IoError on error.
    */
   void open(const String& path);

   /** Gets a dynamic symbol stored in the library.
    * \param symname Name of the symbol to be found.
    * \throw AccessError if the symbol is not found.
    * \return always a valid reference to dynamic code.
    */
   void* getDynSymbol( const String& name ) const;

   /** Gets a dynamic symbol stored in the library (and doesn't throw on error).
    * \param symname Name of the symbol to be found.
    * \return A reference to dynamic code, or 0 on not found.
    *
    * \note This is the system dependent part of getDynSymbol and goes in a
    * system-dependent source file.
    */
   void* getDynSymbol_nothrow( const String& name ) const;


   /** Closes the dynamic library.
    * \throw IoError on error.
    */
   void close();

   FALCON_REFERENCECOUNT_DECLARE_INCDEC(DynLibrary)

   const String& path() const { return m_path; }

private:
   void* m_sysData;
   String m_path;

   virtual ~DynLibrary();


   // disable evil copy constructor.
   DynLibrary( const DynLibrary& )
   {}

   // system dep. part of open() -- must fill m_sysData
   void open_sys(const String& path);

   // system dep. part of close()
   void close_sys();
};


/** Native shared object based module loader.
 */
class FALCON_DYN_CLASS DynLoader
{
public:
   DynLoader();
   virtual ~DynLoader();

   /** Loads a pre-compiled module from a data stream. 
    \param filePath The path where the shared object is stored.
    \param local_name The name under which the module is internally known.
    \return A valid module.
    \throw IOError on load error.
    
    \TODO Use a URI
    */
   Module* load( const String& filePath, const String& local_name );
   
   /** Returns a System-specific extension. */
   static const String& sysExtension();

private:

   Module* load_sys( const String& filePath );
};

}

#endif	/* _FALCON_DYNLOADER_H_ */

/* end of dynloader.h */
