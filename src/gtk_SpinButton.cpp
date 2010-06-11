/**
 *  \file gtk_SpinButton.cpp
 */

#include "gtk_SpinButton.hpp"

#include "gtk_Adjustment.hpp"
#include "gtk_CellEditable.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void SpinButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_SpinButton = mod->addClass( "GtkSpinButton", &SpinButton::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkEntry" ) );
    c_SpinButton->getClassDef()->addInheritance( in );

    c_SpinButton->setWKS( true );
    c_SpinButton->getClassDef()->factory( &SpinButton::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_change_value",&SpinButton::signal_change_value },
    //{ "signal_input",       &SpinButton::signal_input },
    { "signal_output",      &SpinButton::signal_output },
    { "signal_value_changed",&SpinButton::signal_value_changed },
    { "signal_wrapped",     &SpinButton::signal_wrapped },
    { "configure",          &SpinButton::configure },
    { "new_with_range",     &SpinButton::new_with_range },
    { "set_adjustment",     &SpinButton::set_adjustment },
    { "get_adjustment",     &SpinButton::get_adjustment },
    { "set_digits",         &SpinButton::set_digits },
    { "set_increments",     &SpinButton::set_increments },
    { "set_range",          &SpinButton::set_range },
    //{ "get_value_as_float", &SpinButton::get_value_as_float },
    { "get_value_as_int",   &SpinButton::get_value_as_int },
    { "set_value",          &SpinButton::set_value },
    { "set_update_policy",  &SpinButton::set_update_policy },
    { "set_numeric",        &SpinButton::set_numeric },
    { "spin",               &SpinButton::spin },
    { "set_wrap",           &SpinButton::set_wrap },
    { "set_snap_to_ticks",  &SpinButton::set_snap_to_ticks },
    { "update",             &SpinButton::update },
    { "get_digits",         &SpinButton::get_digits },
    { "get_increments",     &SpinButton::get_increments },
    { "get_numeric",        &SpinButton::get_numeric },
    { "get_range",          &SpinButton::get_range },
    { "get_snap_to_ticks",  &SpinButton::get_snap_to_ticks },
    { "get_update_policy",  &SpinButton::get_update_policy },
    { "get_value",          &SpinButton::get_value },
    { "get_wrap",           &SpinButton::get_wrap },
    { NULL, NULL }
    };

    for( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_SpinButton, meth->name, meth->cb );

    Gtk::CellEditable::clsInit( mod, c_SpinButton );
}


SpinButton::SpinButton( const Falcon::CoreClass* gen, const GtkSpinButton* btn )
    :
    Gtk::CoreGObject( gen, (GObject*) btn )
{}


Falcon::CoreObject* SpinButton::factory( const Falcon::CoreClass* gen, void* btn, bool )
{
    return new SpinButton( gen, (GtkSpinButton*) btn );
}


/*#
    @class GtkSpinButton
    @brief Retrieve an integer or floating-point number from the user
    @param adjustment a GtkAdjustment, or nil.
    @param climb_rate the new climb rate.
    @param digits the number of decimal places to display in the spin button.

    A GtkSpinButton is an ideal way to allow the user to set the value of some attribute.
    Rather than having to directly type a number into a GtkEntry, GtkSpinButton allows
    the user to click on one of two arrows to increment or decrement the displayed value.
    A value can still be typed in, with the bonus that it can be checked to ensure it is
    in a given range.

    The main properties of a GtkSpinButton are through a GtkAdjustment.
    See the GtkAdjustment section for more details about an adjustment's properties.

    [...]
*/
FALCON_FUNC SpinButton::init( VMARG )
{
    Item* i_adj = vm->param( 0 );
    Item* i_rate = vm->param( 1 );
    Item* i_digits = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_adj || !( i_adj->isNil() || ( i_adj->isObject()
        && IS_DERIVED( i_adj, GtkAdjustment ) ) )
        || !i_rate || !i_rate->isOrdinal()
        || !i_digits || !i_digits->isInteger() )
        throw_inv_params( "[GtkAdjustment],N,I" );
#endif
    GtkAdjustment* adj = i_adj->isNil() ? NULL
                    : (GtkAdjustment*) COREGOBJECT( i_adj )->getGObject();
    GtkWidget* btn = gtk_spin_button_new(
                    adj, i_rate->forceNumeric(), i_digits->forceInteger() );
    MYSELF;
    self->setGObject( (GObject*) btn );
}


/*#
    @method signal_change_value GtkSpinButton
    @brief Connect a VMSlot to the spinbutton change-value signal and return it
 */
FALCON_FUNC SpinButton::signal_change_value( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    CoreGObject::get_signal( "change_value", (void*) &SpinButton::on_change_value, vm );
}


void SpinButton::on_change_value( GtkSpinButton* btn, GtkScrollType type, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "change_value", "on_change_value", (VMachine*)_vm );
}


#if 0 // todo
FALCON_FUNC SpinButton::signal_input( VMARG );
gint SpinButton::on_input( GtkSpinButton* obj, gpointer arg1, gpointer _vm );
#endif


/*#
    @method signal_output GtkSpinButton
    @brief The output signal can be used to change to formatting of the value that is displayed in the spin buttons entry.
 */
FALCON_FUNC SpinButton::signal_output( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    CoreGObject::get_signal( "output", (void*) &SpinButton::on_output, vm );
}


gboolean SpinButton::on_output( GtkSpinButton* obj, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "output", false );

    if ( !cs || cs->empty() )
        return FALSE; // value not displayed

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_output", it ) )
            {
                printf(
                "[GtkSpinButton::on_output] invalid callback (expected callable)\n" );
                return FALSE; // value not displayed
            }
        }
        vm->callItem( it, 0 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // value displayed
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkSpinButton::on_output] invalid callback (expected boolean)\n" );
            return FALSE; // value not displayed
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // value not displayed
}


/*#
    @method signal_value_changed GtkSpinButton
    @brief Connect a VMSlot to the spinbutton value_changed signal and return it
 */
FALCON_FUNC SpinButton::signal_value_changed( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    CoreGObject::get_signal( "value_changed", (void*) &SpinButton::on_value_changed, vm );
}


void SpinButton::on_value_changed( GtkSpinButton* btn, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "value_changed", "on_value_changed", (VMachine*)_vm );
}


/*#
    @method signal_wrapped GtkSpinButton
    @brief The wrapped signal is emitted right after the spinbutton wraps from its maximum to minimum value or vice-versa.
 */
FALCON_FUNC SpinButton::signal_wrapped( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    CoreGObject::get_signal( "wrapped", (void*) &SpinButton::on_wrapped, vm );
}


void SpinButton::on_wrapped( GtkSpinButton* btn, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "wrapped", "on_wrapped", (VMachine*)_vm );
}


/*#
    @method configure GtkSpinButton
    @brief Changes the properties of an existing spin button.
    @param adjustment a GtkAdjustment, or nil.
    @param climb_rate the new climb rate.
    @param digits the number of decimal places to display in the spin button.

    The adjustment, climb rate, and number of decimal places are all changed
    accordingly, after this function call.
 */
FALCON_FUNC SpinButton::configure( VMARG )
{
    Item* i_adj = vm->param( 0 );
    Item* i_rate = vm->param( 1 );
    Item* i_digits = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_adj || !( i_adj->isNil() || ( i_adj->isObject()
        && IS_DERIVED( i_adj, GtkAdjustment ) ) )
        || !i_rate || !i_rate->isOrdinal()
        || !i_digits || !i_digits->isInteger() )
        throw_inv_params( "[GtkAdjustment],N,I" );
#endif
    GtkAdjustment* adj = i_adj->isNil() ? NULL
                    : (GtkAdjustment*) COREGOBJECT( i_adj )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_spin_button_configure( (GtkSpinButton*)_obj,
                               adj, i_rate->forceNumeric(), i_digits->forceInteger() );
}


/*#
    @method new_with_range GtkSpinButton
    @brief This is a convenience constructor that allows creation of a numeric GtkSpinButton without manually creating an adjustment.
    @param min Minimum allowable value
    @param max Maximum allowable value
    @param step Increment added or subtracted by spinning the widget
    @return The new spin button.

    The value is initially set to the minimum value and a page increment of
    10 * step is the default. The precision of the spin button is equivalent
    to the precision of step.

    Note that the way in which the precision is derived works best if step is
    a power of ten. If the resulting precision is not suitable for your needs,
    use gtk_spin_button_set_digits() to correct it.
 */
FALCON_FUNC SpinButton::new_with_range( VMARG )
{
    Item* i_min = vm->param( 0 );
    Item* i_max = vm->param( 1 );
    Item* i_step = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_min || !i_min->isOrdinal()
        || !i_max || !i_max->isOrdinal()
        || !i_step || !i_step->isOrdinal() )
        throw_inv_params( "N,N,N" );
#endif
    GtkWidget* btn = gtk_spin_button_new_with_range(
                    i_min->forceNumeric(), i_max->forceNumeric(), i_step->forceNumeric() );
    vm->retval( new Gtk::SpinButton( vm->findWKI( "GtkSpinButton" )->asClass(),
                                     (GtkSpinButton*) btn ) );
}


/*#
    @method set_adjustment GtkSpinButton
    @brief Replaces the GtkAdjustment associated with spin_button.
    @param adjustment a GtkAdjustment to replace the existing adjustment
 */
FALCON_FUNC SpinButton::set_adjustment( VMARG )
{
    Item* i_adj = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_adj || !( i_adj->isObject() && IS_DERIVED( i_adj, GtkAdjustment ) ) )
        throw_inv_params( "GtkAdjustment" );
#endif
    GtkAdjustment* adj = (GtkAdjustment*) COREGOBJECT( i_adj )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_spin_button_set_adjustment( (GtkSpinButton*)_obj, adj );
}


/*#
    @method get_adjustment GtkSpinButton
    @brief Get the adjustment associated with a GtkSpinButton
    @return the GtkAdjustment of spin_button
 */
FALCON_FUNC SpinButton::get_adjustment( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GtkAdjustment* adj = gtk_spin_button_get_adjustment( (GtkSpinButton*)_obj );
    vm->retval( new Gtk::Adjustment( vm->findWKI( "GtkAdjustment" )->asClass(), adj ) );
}


/*#
    @method set_digits GtkSpinButton
    @brief Set the precision to be displayed by spin_button. Up to 20 digit precision is allowed.
    @param digits the number of digits after the decimal point to be displayed for the spin button's value
*/
FALCON_FUNC SpinButton::set_digits( VMARG )
{
    Item* i_digits = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if( !i_digits || !i_digits->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_spin_button_set_digits( (GtkSpinButton*)_obj, i_digits->asInteger() );
}


/*#
    @method set_increments GtkSpinButton
    @brief Sets the step and page increments for spin_button.
    @param step increment applied for a button 1 press.
    @param page increment applied for a button 2 press.

    This affects how quickly the value changes when the spin button's arrows are activated.
*/
FALCON_FUNC SpinButton::set_increments( VMARG )
{
    Item* i_step = vm->param( 0 );
    Item* i_page = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if( !i_step || !i_step->isOrdinal()
        || !i_page || !i_page->isOrdinal() )
        throw_inv_params( "N,N" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_spin_button_set_increments( (GtkSpinButton*)_obj,
                                    i_step->forceNumeric(), i_page->forceNumeric() );
}


/*#
    @method set_range GtkSpinButton
    @brief Sets the minimum and maximum allowable values for spin_button
    @param min minimum allowable value
    @param max maximum allowable value
*/
FALCON_FUNC SpinButton::set_range( VMARG )
{
    Item* i_min = vm->param( 0 );
    Item* i_max = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if( !i_min || !i_min->isOrdinal() ||
        !i_max || !i_max->isOrdinal() )
        throw_inv_params( "N,N" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_spin_button_set_range( (GtkSpinButton*)_obj,
                               i_min->forceNumeric(), i_max->forceNumeric() );
}


/*#
    @method get_value_as_int GtkSpinButton
    @brief Get the value spin_button represented as an integer.
    @return the value of spin_button
*/
FALCON_FUNC SpinButton::get_value_as_int( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_spin_button_get_value_as_int( (GtkSpinButton*)_obj ) );
}


/*#
    @method set_value GtkSpinButton
    @brief Set the value of spin_button.
    @param value the new value
*/
FALCON_FUNC SpinButton::set_value( VMARG )
{
    Item* i_value = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if( !i_value || !i_value->isOrdinal() )
        throw_inv_params( "N" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_spin_button_set_value( (GtkSpinButton*)_obj, i_value->forceNumeric() );
}


/*#
    @method set_update_policy GtkSpinButton
    @brief Sets the update behavior of a spin button.
    @param policy A GtkSpinButtonUpdatePolicy value

    This determines whether the spin button is always updated or only when a valid value is set.
*/
FALCON_FUNC SpinButton::set_update_policy( VMARG )
{
    Item* i_policy = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if( !i_policy || !i_policy->isInteger() )
        throw_inv_params( "GtkSpinButtonUpdatePolicy" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_spin_button_set_update_policy( (GtkSpinButton*)_obj,
                                       (GtkSpinButtonUpdatePolicy) i_policy->asInteger() );
}


/*#
    @method set_numeric GtkSpinButton
    @brief Sets the flag that determines if non-numeric text can be typed into the spin button.
    @param numeric flag indicating if only numeric entry is allowed.
*/
FALCON_FUNC SpinButton::set_numeric( VMARG )
{
    Item* i_numeric = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if( !i_numeric || !i_numeric->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_spin_button_set_numeric( (GtkSpinButton*)_obj,
                                 (gboolean) i_numeric->asBoolean() );
}


/*#
    @method spin GtkSpinButton
    @brief Increment or decrement a spin button's value in a specified direction by a specified amount.
    @param direction a GtkSpinType indicating the direction to spin.
    @param increment step increment to apply in the specified direction.
*/
FALCON_FUNC SpinButton::spin( VMARG )
{
    Item* i_direction = vm->param( 0 );
    Item* i_increment = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if( !i_direction || !i_direction->isInteger() ||
        !i_increment || !i_increment->isOrdinal() )
        throw_inv_params( "GtkSpinType,N" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_spin_button_spin( (GtkSpinButton*)_obj,
                          (GtkSpinType) i_direction->asInteger(),
                          i_increment->forceNumeric() );
}


/*#
    @method set_wrap GtkSpinButton
    @brief Sets the flag that determines if a spin button value wraps around to the opposite limit when the upper or lower limit of the range is exceeded.
    @param wrap a flag indicating if wrapping behavior is performed.
*/
FALCON_FUNC SpinButton::set_wrap( VMARG )
{
    Item* i_wrap = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if( !i_wrap || !i_wrap->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_spin_button_set_wrap( (GtkSpinButton*)_obj,
                              (gboolean) i_wrap->asBoolean() );
}


/*#
    @method set_snap_to_ticks GtkSpinButton
    @brief Sets the policy as to whether values are corrected to the nearest step increment when a spin button is activated after providing an invalid value.
    @param snap_to_ticks a flag indicating if invalid values should be corrected.
*/
FALCON_FUNC SpinButton::set_snap_to_ticks( VMARG )
{
    Item* i_snap_to_ticks = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if( !i_snap_to_ticks || !i_snap_to_ticks->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_spin_button_set_snap_to_ticks( (GtkSpinButton*)_obj,
                                       (gboolean) i_snap_to_ticks->asBoolean() );
}


/*#
    @method update GtkSpinButton
    @brief Manually force an update of the spin button.
*/
FALCON_FUNC SpinButton::update( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_spin_button_update( (GtkSpinButton*)_obj );
}


/*#
    @method get_digits GtkSpinButton
    @brief Fetches the precision of spin_button.
    @return the current precision

    See gtk_spin_button_set_digits().
*/
FALCON_FUNC SpinButton::get_digits( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (Falcon::int64) gtk_spin_button_get_digits( (GtkSpinButton*)_obj ) );
}


/*#
    @method get_increments GtkSpinButton
    @brief Gets the current step and page the increments used by spin_button.
    @return An array [ step increment, page increment ]

    See gtk_spin_button_set_increments().
*/
FALCON_FUNC SpinButton::get_increments( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gdouble step, page;
    gtk_spin_button_get_increments( (GtkSpinButton*)_obj, &step, &page );
    CoreArray* result = new CoreArray( 2 );
    result->append( (Falcon::numeric) step );
    result->append( (Falcon::numeric) page );
    vm->retval( result );
}


/*#
    @method get_numeric GtkSpinButton
    @brief Returns whether non-numeric text can be typed into the spin button.
    @return TRUE if only numeric text can be entered

    See gtk_spin_button_set_numeric().
*/
FALCON_FUNC SpinButton::get_numeric( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_spin_button_get_numeric( (GtkSpinButton*)_obj ) );
}


/*#
    @method get_range GtkSpinButton
    @briefGets the range allowed for spin_button.
    @return An array [ min range, max range ]

    See gtk_spin_button_set_range().
*/
FALCON_FUNC SpinButton::get_range( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gdouble min, max;
    gtk_spin_button_get_range( (GtkSpinButton*)_obj, &min, &max );
    CoreArray* result = new CoreArray( 2 );
    result->append( (Falcon::numeric) min );
    result->append( (Falcon::numeric) max );
    vm->retval( result );
}


/*#
    @method get_snap_to_ticks GtkSpinButton
    @brief Returns whether the values are corrected to the nearest step.
    @return TRUE if values are snapped to the nearest step.

    See gtk_spin_button_set_snap_to_ticks().
*/
FALCON_FUNC SpinButton::get_snap_to_ticks( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_spin_button_get_snap_to_ticks( (GtkSpinButton*)_obj ) );
}


/*#
    @method get_update_policy GtkSpinButton
    @brief Gets the update behavior of a spin button. See gtk_spin_button_set_update_policy().
    @return the current update policy (GtkSpinButtonUpdatePolicy)
*/
FALCON_FUNC SpinButton::get_update_policy( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (Falcon::int64) gtk_spin_button_get_update_policy( (GtkSpinButton*)_obj ) );
}


/*#
    @method get_value GtkSpinButton
    @brief Get the value in the spin_button.
    @return the value of spin_button
*/
FALCON_FUNC SpinButton::get_value( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (Falcon::numeric) gtk_spin_button_get_value( (GtkSpinButton*)_obj ) );
}


/*#
    @method get_wrap GtkSpinButton
    @briefReturns whether the spin button's value wraps around to the opposite limit when the upper or lower limit of the range is exceeded. See gtk_spin_button_set_wrap().
    @return TRUE if the spin button wraps around
*/
FALCON_FUNC SpinButton::get_wrap( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_spin_button_get_wrap( (GtkSpinButton*)_obj ) );
}


} //Gtk
} //Falcon
