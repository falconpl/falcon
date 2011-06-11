/**
 *  \file gtk_VPaned.cpp
 */

#include "gtk_VPaned.hpp"

#include <gtk/gtk.h>

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void VPaned::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_VPaned = mod->addClass( "GtkVPaned", &VPaned::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkPaned" ) );
    c_VPaned->getClassDef()->addInheritance( in );

    c_VPaned->getClassDef()->factory( &VPaned::factory );
}


VPaned::VPaned( const Falcon::CoreClass* gen, const GtkVPaned* paned )
    :
    Gtk::CoreGObject( gen, (GObject*) paned )
{}


Falcon::CoreObject* VPaned::factory( const Falcon::CoreClass* gen, void* paned, bool )
{
    return new VPaned( gen, (GtkVPaned*) paned );
}


/*#
    @class GtkVPaned
    @brief A container with two panes arranged vertically

    The VPaned widget is a container widget with two children arranged vertically.
    The division between the two panes is adjustable by the user by dragging a handle.
    See GtkPaned for details.
 */
FALCON_FUNC VPaned::init( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GtkWidget* wdt = gtk_vpaned_new();
    self->setObject( (GObject*) wdt );
}


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
