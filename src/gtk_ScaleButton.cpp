/**
 *  \file gtk_ScaleButton.cpp
 */

#include "gtk_ScaleButton.hpp"

#include "gtk_Adjustment.hpp"
#include "gtk_Orientable.hpp"
#include "gtk_Widget.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void ScaleButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_ScaleButton = mod->addClass( "GtkScaleButton", &ScaleButton::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkButton" ) );
    c_ScaleButton->getClassDef()->addInheritance( in );

    c_ScaleButton->setWKS( true );
    c_ScaleButton->getClassDef()->factory( &ScaleButton::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_popdown",     &ScaleButton::signal_popdown },
    { "signal_popup",       &ScaleButton::signal_popup },
    { "signal_value_changed",&ScaleButton::signal_value_changed },
    { "set_adjustment",     &ScaleButton::set_adjustment },
    { "set_icons",          &ScaleButton::set_icons },
    { "set_value",          &ScaleButton::set_value },
    { "get_adjustment",     &ScaleButton::get_adjustment },
    { "get_value",          &ScaleButton::get_value },
#if GTK_CHECK_VERSION( 2, 14, 0 )
    { "get_popup",          &ScaleButton::get_popup },
    { "get_plus_button",    &ScaleButton::get_plus_button },
    { "get_minus_button",   &ScaleButton::get_minus_button },
#if 0 // deprecated
    { "set_orientation",    &ScaleButton::set_orientation },
    { "set_orientation",    &ScaleButton::set_orientation },
#endif
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_ScaleButton, meth->name, meth->cb );

#if GTK_CHECK_VERSION( 2, 16, 0 )
    Gtk::Orientable::clsInit( mod, c_ScaleButton );
#endif
}


ScaleButton::ScaleButton( const Falcon::CoreClass* gen, const GtkScaleButton* btn )
    :
    Gtk::CoreGObject( gen, (GObject*) btn )
{}


Falcon::CoreObject* ScaleButton::factory( const Falcon::CoreClass* gen, void* btn, bool )
{
    return new ScaleButton( gen, (GtkScaleButton*) btn );
}


/*#
    @class GtkScaleButton
    @brief A button which pops up a scale
    @param size a stock icon size (GtkIconSize).
    @param min the minimum value of the scale (usually 0)
    @param max the maximum value of the scale (usually 100)
    @param step the stepping of value when a scroll-wheel event, or up/down arrow event occurs (usually 2)
    @param icons an array of icon names (strings), or nil if you want to set the list later with set_icons().

    GtkScaleButton provides a button which pops up a scale widget. This kind of
    widget is commonly used for volume controls in multimedia applications, and
    GTK+ provides a GtkVolumeButton subclass that is tailored for this use case.
 */
FALCON_FUNC ScaleButton::init( VMARG )
{
    MYSELF;
    if ( self->getGObject() )
        return;

    Gtk::ArgCheck0 args( vm, "GtkIconSize,N,N,N,[A]" );
    int size = args.getInteger( 0 );
    gdouble min = args.getNumeric( 1 );
    gdouble max = args.getNumeric( 2 );
    gdouble step = args.getNumeric( 3 );
    CoreArray* a_icons = args.getArray( 4, false );

    GtkWidget* wdt;

    if ( a_icons && a_icons->length() )
    {
        gchar* cstr;
        AutoCString* tmp;
        getGCharArray( a_icons, &cstr, &tmp );
        wdt = gtk_scale_button_new(
                (GtkIconSize) size, min, max, step, (const gchar**) cstr );
        memFree( cstr );
        memFree( tmp );
    }
    else
        wdt = gtk_scale_button_new( (GtkIconSize) size, min, max, step, NULL );

    self->setGObject( (GObject*) wdt );
}


/*#
    @method signal_popdown GtkScaleButton
    @brief The popdown signal is a keybinding signal which gets emitted to popdown the scale widget.

    The default binding for this signal is Escape.
 */
FALCON_FUNC ScaleButton::signal_popdown( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "popdown", (void*) &ScaleButton::on_popdown, vm );
}


void ScaleButton::on_popdown( GtkScaleButton* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "popdown", "on_popdown", (VMachine*)_vm );
}


/*#
    @method signal_popup GtkScaleButton
    @brief The popup signal is a keybinding signal which gets emitted to popup the scale widget.

    The default bindings for this signal are Space, Enter and Return.
 */
FALCON_FUNC ScaleButton::signal_popup( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "popup", (void*) &ScaleButton::on_popup, vm );
}


void ScaleButton::on_popup( GtkScaleButton* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "popup", "on_popup", (VMachine*)_vm );
}


/*#
    @method signal_value_changed GtkScaleButton
    @brief The value-changed signal is emitted when the value field has changed.
 */
FALCON_FUNC ScaleButton::signal_value_changed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "value_changed", (void*) &ScaleButton::on_value_changed, vm );
}


void ScaleButton::on_value_changed( GtkScaleButton* obj, gdouble value, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "value_changed", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_value_changed", it ) )
            {
                printf(
                "[GtkScaleButton::on_value_changed] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( value );
        vm->callItem( it, 1 );
    }
    while ( iter.hasCurrent() );
}


/*#
    @method set_adjustment GtkScaleButton
    @brief Sets the GtkAdjustment to be used as a model for the GtkScaleButton's scale.
    @param adjustment a GtkAdjustment
 */
FALCON_FUNC ScaleButton::set_adjustment( VMARG )
{
    Item* i_adj = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_adj || i_adj->isNil() || !i_adj->isObject()
        || !IS_DERIVED( i_adj, GtkAdjustment ) )
        throw_inv_params( "GtkAdjustment" );
#endif
    GtkAdjustment* adj = (GtkAdjustment*) COREGOBJECT( i_adj )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_scale_button_set_adjustment( (GtkScaleButton*)_obj, adj );
}


/*#
    @method set_icons GtkScaleButton
    @brief Sets the icons to be used by the scale button.
    @param icons an array of icon names (strings).
 */
FALCON_FUNC ScaleButton::set_icons( VMARG )
{
    Item* i_arr = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_arr || !i_arr->isArray() )
        throw_inv_params( "A" );
#endif
    CoreArray* arr = i_arr->asArray();
    MYSELF;
    GET_OBJ( self );
    if ( arr->length() )
    {
        gchar* cstr;
        AutoCString* tmp;
        getGCharArray( arr, &cstr, &tmp );
        gtk_scale_button_set_icons( (GtkScaleButton*)_obj, (const gchar**) &cstr );
        memFree( cstr );
        memFree( tmp );
    }
    else
        gtk_scale_button_set_icons( (GtkScaleButton*)_obj, NULL );
}


/*#
    @method set_value GtkScaleButton
    @brief Sets the current value of the scale.
    @param value new value of the scale button

    If the value is outside the minimum or maximum range values, it will be
    clamped to fit inside them. The scale button emits the "value-changed"
    signal if the value changes.
 */
FALCON_FUNC ScaleButton::set_value( VMARG )
{
    Item* i_val = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_val || !i_val->isOrdinal() )
        throw_inv_params( "N" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_scale_button_set_value( (GtkScaleButton*)_obj, i_val->asNumeric() );
}


/*#
    @method get_adjustment GtkScaleButton
    @brief Gets the GtkAdjustment associated with the GtkScaleButton's scale.
    @return the adjustment associated with the scale
 */
FALCON_FUNC ScaleButton::get_adjustment( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkAdjustment* adj = gtk_scale_button_get_adjustment( (GtkScaleButton*)_obj );
    vm->retval( new Gtk::Adjustment( vm->findWKI( "GtkAdjustment" )->asClass(), adj ) );
}


/*#
    @method get_value GtkScaleButton
    @brief Gets the current value of the scale button.
    @return current value of the scale button
 */
FALCON_FUNC ScaleButton::get_value( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_scale_button_get_value( (GtkScaleButton*)_obj ) );
}


#if GTK_CHECK_VERSION( 2, 14, 0 )
/*#
    @method get_popup GtkScaleButton
    @brief Retrieves the popup of the GtkScaleButton.
    @return the popup of the GtkScaleButton (as a GtkWidget).
 */
FALCON_FUNC ScaleButton::get_popup( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_scale_button_get_popup( (GtkScaleButton*)_obj );
    vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
}


/*#
    @method get_plus_button GtkScaleButton
    @brief Retrieves the plus button of the GtkScaleButton.
    @return the plus button of the GtkScaleButton (as a GtkWidget).
 */
FALCON_FUNC ScaleButton::get_plus_button( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_scale_button_get_plus_button( (GtkScaleButton*)_obj );
    vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
}


/*#
    @method get_minus_button GtkScaleButton
    @brief Retrieves the minus button of the GtkScaleButton.
    @return the minus button of the GtkScaleButton (as a GtkWidget).
 */
FALCON_FUNC ScaleButton::get_minus_button( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_scale_button_get_minus_button( (GtkScaleButton*)_obj );
    vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
}


#if 0 // deprecated
FALCON_FUNC ScaleButton::set_orientation( VMARG );
FALCON_FUNC ScaleButton::get_orientation( VMARG );
#endif

#endif // GTK_CHECK_VERSION( 2, 14, 0 )


} // Gtk
} // Falcon
