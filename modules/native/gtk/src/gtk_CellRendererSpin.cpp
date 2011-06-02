/**
 *  \file gtk_CellRendererSpin.cpp
 */

#include "gtk_CellRendererSpin.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void CellRendererSpin::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_CellRendererSpin = mod->addClass( "GtkCellRendererSpin", &CellRendererSpin::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkCellRendererText" ) );
    c_CellRendererSpin->getClassDef()->addInheritance( in );

    //c_CellRendererSpin->setWKS( true );
    c_CellRendererSpin->getClassDef()->factory( &CellRendererSpin::factory );

}


CellRendererSpin::CellRendererSpin( const Falcon::CoreClass* gen, const GtkCellRendererSpin* renderer )
    :
    Gtk::CoreGObject( gen, (GObject*) renderer )
{}


Falcon::CoreObject* CellRendererSpin::factory( const Falcon::CoreClass* gen, void* renderer, bool )
{
    return new CellRendererSpin( gen, (GtkCellRendererSpin*) renderer );
}


/*#
    @class GtkCellRendererSpin
    @brief Renders a spin button in a cell

    GtkCellRendererSpin renders text in a cell like GtkCellRendererText from
    which it is derived. But while GtkCellRendererText offers a simple entry to
    edit the text, GtkCellRendererSpin offers a GtkSpinButton widget. Of course
    that means that the text has to be parseable as a floating point number.

    The range of the spinbutton is taken from the adjustment property of the
    cell renderer, which can be set explicitly or mapped to a column in the tree
    model, like all properties of cell renders. GtkCellRendererSpin also has
    properties for the climb rate and the number of digits to display. Other
    GtkSpinButton properties can be set in a handler for the start-editing signal.
 */
FALCON_FUNC CellRendererSpin::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setObject( (GObject*) gtk_cell_renderer_spin_new() );
}


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
