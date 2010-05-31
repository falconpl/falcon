/**
 *  \file gtk_VolumeButton.cpp
 */

#include "gtk_VolumeButton.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void VolumeButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_VolumeButton = mod->addClass( "GtkVolumeButton", &VolumeButton::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkScaleButton" ) );
    c_VolumeButton->getClassDef()->addInheritance( in );

    c_VolumeButton->getClassDef()->factory( &VolumeButton::factory );
}


VolumeButton::VolumeButton( const Falcon::CoreClass* gen, const GtkVolumeButton* btn )
    :
    Gtk::CoreGObject( gen, (GObject*) btn )
{}


Falcon::CoreObject* VolumeButton::factory( const Falcon::CoreClass* gen, void* btn, bool )
{
    return new VolumeButton( gen, (GtkVolumeButton*) btn );
}


/*#
    @class GtkVolumeButton
    @brief A button which pops up a volume control

    GtkVolumeButton is a subclass of GtkScaleButton that has been tailored for use
    as a volume control widget with suitable icons, tooltips and accessible labels.

    The constructor creates a GtkVolumeButton, with a range between 0.0 and 1.0,
    with a stepping of 0.02. Volume values can be obtained and modified using the
    functions from GtkScaleButton.
 */
FALCON_FUNC VolumeButton::init( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    self->setGObject( (GObject*) gtk_volume_button_new() );
}


} // Gtk
} // Falcon
