import { setPersonModels, useRack } from '../../stores/rackSelection';
import ModelSelect, { type RackOption } from './ModelSelect';

type Props = { personKey: string; label: string; swapperOptions: RackOption[]; enhancerOptions: RackOption[] };

/** Pins a swapper and an enhancer to one detected person. "Scene default" = no override. */
export default function PersonModelPicker({ personKey, label, swapperOptions, enhancerOptions }: Props) {
  const models = useRack(state => state.perPerson[personKey]);
  const withDefault = (options: RackOption[]): RackOption[] => [{ id: '', modelId: null, label: 'Scene default' }, ...options.filter(option => option.id !== 'none'), ...options.filter(option => option.id === 'none')];
  return (
    <div className="rack-person" data-testid={`rack-person-${personKey}`} role="group" aria-label={`Models for ${label}`}>
      <b>{label}</b>
      <ModelSelect label="Swapper" category="swapper" value={models?.swapper ?? ''} onChange={id => setPersonModels(personKey, { swapper: id })} options={withDefault(swapperOptions)} includeRegistered={false} />
      <ModelSelect label="Enhancer" category="enhancer" value={models?.enhancer ?? ''} onChange={id => setPersonModels(personKey, { enhancer: id })} options={withDefault(enhancerOptions)} includeRegistered={false} />
    </div>
  );
}
