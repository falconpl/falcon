/*
   FALCON - The Falcon Programming Language.
   FILE: modmap.h

   A simple class orderly guarding modules.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 09 Aug 2011 00:43:47 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_MODMAP_H_
#define _FALCON_MODMAP_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/loadmode.h>

namespace Falcon 
{

class Module;
class ModGroup;

/** A simple class orderly guarding modules.
 */
class FALCON_DYN_CLASS ModMap
{
public:
   
   /** An entry of a the module map.
    An entry in a module map is composed of:
    - The pointer to the module;
    - The ownership status (whether the map can destroy the module or not);
    - The module load mode (load, import private or import public).
    */
      
   class Entry
   {
   public:
      
      Entry( Module* mod, t_loadMode im, bool bOwn ):
         m_module(mod),
         m_imode(im),
         m_bOwn( bOwn )
      {}
      
      ~Entry();
      
      Module* module() const { return m_module; }
      bool own() const { return m_bOwn; }
      t_loadMode imode() const { return m_imode; }
      void imode( t_loadMode eml ) { m_imode = eml; }
      
   private:
      Module* m_module;
      t_loadMode m_imode;
      bool m_bOwn;
   };
  
   ModMap();
   ~ModMap();
   
   void add( Module* mod, t_loadMode im, bool bown = true );
   void remove( Module* mod );
   
   Entry* findByURI( const String& path ) const;
   Entry* findByName( const String& name ) const;
   Entry* findByModule( Module* mod ) const;
   

   class EntryEnumerator 
   {
   public:
      virtual ~EntryEnumerator()=0;
      virtual void operator()( Entry* e );
   };
   
   void enumerate( EntryEnumerator& rator ) const;
   
   bool empty() const;
private:
   class Private;
   Private* _p;
   
   friend class ModGroup;
};

}

#endif	/* _FALCON_MODMAP_H_ */

/* end of modmap.h */
