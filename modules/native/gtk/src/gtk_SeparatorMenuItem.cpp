/**
 *  \file gtk_SeparatorMenuItem.cpp
 */

#include "gtk_SeparatorMenuItem.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void SeparatorMenuItem::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_SeparatorMenuItem = mod->addClass( "GtkSeparatorMenuItem", &SeparatorMenuItem::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkMenuItem" ) );
    c_SeparatorMenuItem->getClassDef()->addInheritance( in );

    //c_SeparatorMenuItem->setWKS( true );
    c_SeparatorMenuItem->getClassDef()->factory( &SeparatorMenuItem::factory );
}


SeparatorMenuItem::SeparatorMenuItem( const Falcon::CoreClass* gen, const GtkSeparatorMenuItem* itm )
    :
    Gtk::CoreGObject( gen, (GObject*) itm )
{}


Falcon::CoreObject* SeparatorMenuItem::factory( const Falcon::CoreClass* gen, void* itm, bool )
{
    return new SeparatorMenuItem( gen, (GtkSeparatorMenuItem*) itm );
}


/*#
    @class GtkSeparatorMenuItem
    @brief A separator used in menus

    The GtkSeparatorMenuItem is a separator used to group items within a menu.
    It displays a horizontal line with a shadow to make it appear sunken into
    the interface.
 */
FALCON_FUNC SeparatorMenuItem::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setObject( (GObject*) gtk_separator_menu_item_new() );
}


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
