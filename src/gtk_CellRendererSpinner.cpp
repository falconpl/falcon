/**
 *  \file gtk_CellRendererSpinner.cpp
 */

#include "gtk_CellRendererSpinner.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void CellRendererSpinner::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_CellRendererSpinner = mod->addClass( "GtkCellRendererSpinner", &CellRendererSpinner::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkCellRenderer" ) );
    c_CellRendererSpinner->getClassDef()->addInheritance( in );

    //c_CellRendererSpinner->setWKS( true );
    c_CellRendererSpinner->getClassDef()->factory( &CellRendererSpinner::factory );

}


CellRendererSpinner::CellRendererSpinner( const Falcon::CoreClass* gen, const GtkCellRendererSpinner* renderer )
    :
    Gtk::CoreGObject( gen, (GObject*) renderer )
{}


Falcon::CoreObject* CellRendererSpinner::factory( const Falcon::CoreClass* gen, void* renderer, bool )
{
    return new CellRendererSpinner( gen, (GtkCellRendererSpinner*) renderer );
}


/*#
    @class GtkCellRendererSpinner
    @brief Renders a spinning animation in a cell

    GtkCellRendererSpinner renders a spinning animation in a cell, very similar
    to GtkSpinner. It can often be used as an alternative to a GtkCellRendererProgress
    for displaying indefinite activity, instead of actual progress.

    To start the animation in a cell, set the "active" property to TRUE and
    increment the "pulse" property at regular intervals. The usual way to set
    the cell renderer properties for each cell is to bind them to columns in your
    tree model using e.g. gtk_tree_view_column_add_attribute().
 */
FALCON_FUNC CellRendererSpinner::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setGObject( (GObject*) gtk_cell_renderer_spinner_new() );
}


} // Gtk
} // Falcon
