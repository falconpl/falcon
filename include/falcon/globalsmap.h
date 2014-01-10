/*
   FALCON - The Falcon Programming Language.
   FILE: globals.h

   Map holding variables and associated data for global storage.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 08 Jan 2013 18:36:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_VARDATAMAP_H_
#define _FALCON_VARDATAMAP_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/item.h>
#include <falcon/variable.h>

namespace Falcon {

class String;
class Module;

/** Map holding variables and associated data for global storage in modules.
 *
 */
class FALCON_DYN_CLASS GlobalsMap
{
public:
   GlobalsMap();
   ~GlobalsMap();

   class Data
   {
   public:
      Data():
         m_bExtern(false)
      {
         m_storage.setNil();
         m_data = &m_storage;
      }

      Data(const Item& data):
         m_storage( data ),
         m_bExtern(false)
      {
         m_data = &m_storage;
      }

     Data(Item* pdata):
        m_bExtern(true)
     {
        m_storage.setNil();
        m_data = pdata;
     }

      Item* m_data;
      Item m_storage;
      bool m_bExtern;
   };


   /** Adds a global variable, possibly exportable.
      \param name The name of the symbol referencing the variable.
      \param bExport if true, the returned symbol will be exported.
      \param value the value to be added.
      \return A variable containing the global ID or 0 if the variable
      was already declared.

      Creates a variable with a value in the module global vector.

      \note The method won't check the contents of the item; if it's a function or a class,
      it will normally be available as a global module variable, but it won't be added
      to the module mantra map.

      */
   Data* add( const String& name, const Item& value, bool bExport = false );
   Data* add( Symbol* sym, const Item& value, bool bExport = false );

   Data* promote( const String& name, const Item& value, bool bExport = false );
   Data* promote( Symbol* sym, const Item& value, bool bExport = false );

   /** Imports a variable in as external.
    * \param sym The symbol indicating the variable that is to be imported.
    * \param value The new value for that variable.
    * \return the data entry associated with that variable.
    *
    * This method forces creation or update of the given variable,
    * whether it previously existed in the table or not. The variable
    * is then associated to an external pointer, which must stay available for
    * the scope of the owner module existence. This means the value must either
    * come from the engine, from an embedding application or from a module in the
    * same module space of the owner module.
    */
   Data* addExtern( Symbol* sym, Item* value );
   Data* addExtern( const String& symName, Item* value );

   bool remove( const String& name );
   bool remove( Symbol* sym );

  /** Export a previously declared symbol.
   \param name The name of the symbol to be exported.
   \param bAlready will be set to true if the symbol was already exported.
   \return The exported variable or zero if the name isn't known.

   \note In the falcon language it is NOT legal to export undefined symbols -- to avoid mistyping.
   */
   Data* exportGlobal( const String& name, bool &bAlready );

   inline Data* exportGlobal( const String& name ) {
      bool dummy;
      return exportGlobal(name, dummy);
   }

   Data* exportGlobal( Symbol* sym, bool &bAlready );
   Data* exportGlobal( Symbol* sym )
   {
      bool dummy;
      return exportGlobal(sym, dummy);
   }

   /** Finds the value of a global variable by name.
   \param name The variable name to be searched.
   \return A pointer to a global value, or 0 if not found.

   Notice that the returned value might be either defined as a static
   value in this module or being imported from other sources (i.e. the
   exported data in the host module space).

   \note Still unimported extern variables return 0.
   */
   Item* getValue( const String& name ) const;
   Item* getValue( Symbol* sym ) const;

   /** Gets the global data associated with the given variable name.
    * \param name The name of the global variable to be searched.
    * \return A valid Data pointer on success, 0 if the name is unknown.
    *
    * The returned value can be manipulated to change the stored value
    * or the item pointer (i.e. to make it to point to the exported globals
    * in the module space where a module is stored).
    */
   Data* get(const String& name ) const;
   Data* get( Symbol* sym ) const;

   /** Checks if a variable is exported given its name.
   */
   bool isExported( const String& name ) const;
   bool isExported( Symbol* sym ) const;

   /** Mark (dynamic) modules for Garbage Collecting.
    \param mark the current GC mark.

    This is used by ClassModule when the module is handed to the Virtual
    Machine for dynamic collection.
    */
   void gcMark( uint32 mark );

   /** Determines if a module can be reclaimed.
    \return last GC mark.

    This is used by ClassModule when the module is handed to the Virtual
    Machine for dynamic collection.
    */

   uint32 lastGCMark() const;

   class VariableEnumerator
   {
   public:
      virtual void operator() ( Symbol* sym, Item*& value ) = 0;
   };

   /** Enumerate all the variables exported by this module.
    *
    * By definition, exported variables are global and provide a valid pointer
    * when getGlobalValue() is invoked.
    *
    * Notice that the VariableEnumerator receives a value by pointer-reference,
    * allowing the receiver to change the pointer if needed.
    */
   void enumerateExports( VariableEnumerator& rator ) const;
   void enumerate( VariableEnumerator& rator ) const;

   /** Returns the count of global variables. */
   uint32 size() const;

   void flatten( VMContext* ctx, ItemArray& subItems ) const;
   void unflatten( VMContext* ctx, ItemArray& subItems, uint32 start, uint32 &count );

   bool isExportAll() const { return m_bExportAll; }
   void setExportAll( bool bMode ) { m_bExportAll = bMode; }

private:
   class Private;
   GlobalsMap::Private* _p;

   bool m_bExportAll;
};

}

#endif

/* end of globalsmap.h */
